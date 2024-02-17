// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "common.h"
#include "rasterize.h"

//------------------------------------------------------------------------
// Cuda forward rasterizer pixel shader kernel.

__global__ void RasterizeCudaFwdShaderKernel(const RasterizeCudaFwdShaderParams p)
{
    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Fetch triangle idx.
    int triIdx = p.in_idx[pidx] - 1;
    if (triIdx < 0 || triIdx >= p.numTriangles)
    {
        // No or corrupt triangle.
        ((float4*)p.out)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0); // Clear out.
        ((float4*)p.out_db)[pidx] = make_float4(0.0, 0.0, 0.0, 0.0); // Clear out_db.
        return;
    }

    // Fetch vertex indices.
    int vi0 = p.tri[triIdx * 3 + 0];
    int vi1 = p.tri[triIdx * 3 + 1];
    int vi2 = p.tri[triIdx * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // In instance mode, adjust vertex indices by minibatch index.
    if (p.instance_mode)
    {
        vi0 += pz * p.numVertices;
        vi1 += pz * p.numVertices;
        vi2 += pz * p.numVertices;
    }

    // Fetch vertex positions.
    float4 p0 = ((float4*)p.pos)[vi0];
    float4 p1 = ((float4*)p.pos)[vi1];
    float4 p2 = ((float4*)p.pos)[vi2];

    // Evaluate edge functions.
    float fx = p.xs * (float)px + p.xo;
    float fy = p.ys * (float)py + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Perspective correct, normalized barycentrics.
    float iw = 1.f / (a0 + a1 + a2);
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Compute z/w for depth buffer.
    float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
    float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
    float zw = z / w;

    // Clamps to avoid NaNs.
    b0 = __saturatef(b0); // Clamp to [+0.0, 1.0].
    b1 = __saturatef(b1); // Clamp to [+0.0, 1.0].
    zw = fmaxf(fminf(zw, 1.f), -1.f);

    // Emit output.
    ((float4*)p.out)[pidx] = make_float4(b0, b1, zw, (float)(triIdx + 1));

    // Calculate bary pixel differentials.
    float dfxdx = p.xs * iw;
    float dfydy = p.ys * iw;
    float da0dx = p2.y*p1.w - p1.y*p2.w;
    float da0dy = p1.x*p2.w - p2.x*p1.w;
    float da1dx = p0.y*p2.w - p2.y*p0.w;
    float da1dy = p2.x*p0.w - p0.x*p2.w;
    float da2dx = p1.y*p0.w - p0.y*p1.w;
    float da2dy = p0.x*p1.w - p1.x*p0.w;
    float datdx = da0dx + da1dx + da2dx;
    float datdy = da0dy + da1dy + da2dy;
    float dudx = dfxdx * (b0 * datdx - da0dx);
    float dudy = dfydy * (b0 * datdy - da0dy);
    float dvdx = dfxdx * (b1 * datdx - da1dx);
    float dvdy = dfydy * (b1 * datdy - da1dy);

    // Emit bary pixel differentials.
    ((float4*)p.out_db)[pidx] = make_float4(dudx, dudy, dvdx, dvdy);
}


__device__ inline void mv_multiply_4(float* matrix, float* vector, float* res)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            res[i] += matrix[4*i + j] * vector[j];
        }
    }
}


//------------------------------------------------------------------------
// Gradient Cuda kernel.

template <bool ENABLE_DB>
static __forceinline__ __device__ void RasterizeGradKernelTemplate(const RasterizeGradParams p)
{
    // Temporary space for coalesced atomics.
    CA_DECLARE_TEMP(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH * RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT);

    // Calculate pixel position.
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= p.width || py >= p.height || pz >= p.depth)
        return;

    // Pixel index.
    int pidx = px + p.width * (py + p.height * pz);

    // Read triangle idx and dy.
    float2 dy  = ((float2*)p.dy)[pidx * 2];
    float4 ddb = ENABLE_DB ? ((float4*)p.ddb)[pidx] : make_float4(0.f, 0.f, 0.f, 0.f);

    int triIdx = (int)(((float*)p.out)[pidx * 4 + 3]) - 1;

    int object_idx = (int)(((int*)p.out2)[pidx * 4 + 0]) - 1;
    int depth = p.depth;
    float* pose = p.pose + ( depth * 16 * object_idx +  pz * 16);

    // Exit if nothing to do.
    if (triIdx < 0 || triIdx >= p.numTriangles)
        return; // No or corrupt triangle.
    int grad_all_dy = __float_as_int(dy.x) | __float_as_int(dy.y); // Bitwise OR of all incoming gradients.
    int grad_all_ddb = 0;
    if (ENABLE_DB)
        grad_all_ddb = __float_as_int(ddb.x) | __float_as_int(ddb.y) | __float_as_int(ddb.z) | __float_as_int(ddb.w);
    if (((grad_all_dy | grad_all_ddb) << 1) == 0)
        return; // All incoming gradients are +0/-0.

    // Fetch vertex indices.
    int vi0 = p.tri[triIdx * 3 + 0];
    int vi1 = p.tri[triIdx * 3 + 1];
    int vi2 = p.tri[triIdx * 3 + 2];

    // Bail out if vertex indices are corrupt.
    if (vi0 < 0 || vi0 >= p.numVertices ||
        vi1 < 0 || vi1 >= p.numVertices ||
        vi2 < 0 || vi2 >= p.numVertices)
        return;

    // // In instance mode, adjust vertex indices by minibatch index.
    // if (p.instance_mode)
    // {
    //     vi0 += pz * p.numVertices;
    //     vi1 += pz * p.numVertices;
    //     vi2 += pz * p.numVertices;
    // }

    float* proj = p.proj;

    // Initialize coalesced atomics.
    CA_SET_GROUP(triIdx);

    // Apply projection_matrix  * pose[frame_idx, object_index] * p.pos (which is in object space) to get the clip vertex positions.
    // The following computations assume p.pos is in clip space.
    float* vertex_1_object_frame = p.pos + vi0 * 4;
    float* vertex_2_object_frame = p.pos + vi1 * 4;
    float* vertex_3_object_frame = p.pos + vi2 * 4;

    float vertex_1_camera_frame[4] = {0};
    float vertex_2_camera_frame[4] = {0};
    float vertex_3_camera_frame[4] = {0};
    mv_multiply_4(pose, vertex_1_object_frame, vertex_1_camera_frame);
    mv_multiply_4(pose, vertex_2_object_frame, vertex_2_camera_frame);
    mv_multiply_4(pose, vertex_3_object_frame, vertex_3_camera_frame);

    float vertex_1_clip_space[4] = {0};
    float vertex_2_clip_space[4] = {0};
    float vertex_3_clip_space[4] = {0};
    mv_multiply_4(proj, vertex_1_object_frame, vertex_1_clip_space);
    mv_multiply_4(proj, vertex_2_object_frame, vertex_2_clip_space);
    mv_multiply_4(proj, vertex_3_object_frame, vertex_3_clip_space);

    // Fetch vertex positions.
    // float4 p0 = ((float4*)p.pos)[vi0];
    // float4 p1 = ((float4*)p.pos)[vi1];
    // float4 p2 = ((float4*)p.pos)[vi2];
    float4 p0 = ((float4*)vertex_1_clip_space)[0];
    float4 p1 = ((float4*)vertex_2_clip_space)[0];
    float4 p2 = ((float4*)vertex_3_clip_space)[0];

    // Evaluate edge functions.
    float fx = p.xs * (float)px + p.xo;
    float fy = p.ys * (float)py + p.yo;
    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;
    float a0 = p1x*p2y - p1y*p2x;
    float a1 = p2x*p0y - p2y*p0x;
    float a2 = p0x*p1y - p0y*p1x;

    // Compute inverse area with epsilon.
    float at = a0 + a1 + a2;
    float ep = copysignf(1e-6f, at); // ~1 pixel in 1k x 1k image.
    float iw = 1.f / (at + ep);

    // Perspective correct, normalized barycentrics.
    float b0 = a0 * iw;
    float b1 = a1 * iw;

    // Position gradients.
    float gb0  = dy.x * iw;
    float gb1  = dy.y * iw;
    float gbb  = gb0 * b0 + gb1 * b1;
    float gp0x = gbb * (p2y - p1y) - gb1 * p2y;
    float gp1x = gbb * (p0y - p2y) + gb0 * p2y;
    float gp2x = gbb * (p1y - p0y) - gb0 * p1y + gb1 * p0y;
    float gp0y = gbb * (p1x - p2x) + gb1 * p2x;
    float gp1y = gbb * (p2x - p0x) - gb0 * p2x;
    float gp2y = gbb * (p0x - p1x) + gb0 * p1x - gb1 * p0x;
    float gp0w = -fx * gp0x - fy * gp0y;
    float gp1w = -fx * gp1x - fy * gp1y;
    float gp2w = -fx * gp2x - fy * gp2y;

    float loss_grad_v1_clip_space[4] = {gp0x, gp0y, 0.f, gp0w};
    float loss_grad_v2_clip_space[4] = {gp1x, gp1y, 0.f, gp1w};
    float loss_grad_v3_clip_space[4] = {gp2x, gp2y, 0.f, gp2w};

    for (int i = 0; i < 16; i++) {
        int row=i/4, col=i%4;
        //XXX somehow get the xyzw of the vertices
        //col-th coordinate of vi0-, vi1-, vi2-th vertices
        float  vertex_term1 = ((float*) vertex_1_object_frame)[col];
        float  vertex_term2 = ((float*) vertex_2_object_frame)[col];
        float  vertex_term3 = ((float*) vertex_3_object_frame)[col];

        float accumulated_gradient = 0.0;
        accumulated_gradient += loss_grad_v1_clip_space[0] * proj[row] * vertex_term1;
        accumulated_gradient += loss_grad_v2_clip_space[0] * proj[row] * vertex_term2;
        accumulated_gradient += loss_grad_v3_clip_space[0] * proj[row] * vertex_term3;

        accumulated_gradient += loss_grad_v1_clip_space[1] * proj[4 + row] * vertex_term1;
        accumulated_gradient += loss_grad_v2_clip_space[1] * proj[4 + row]* vertex_term2;
        accumulated_gradient += loss_grad_v3_clip_space[1] * proj[4 + row]* vertex_term3;

        accumulated_gradient += loss_grad_v1_clip_space[3] * proj[12 + row] * vertex_term1;
        accumulated_gradient += loss_grad_v2_clip_space[3] * proj[12 + row]* vertex_term2;
        accumulated_gradient += loss_grad_v3_clip_space[3] * proj[12 + row]* vertex_term3;

        // Fix this;
        caAtomicAdd(
            p.grad + (depth * 16 * object_idx +  pz * 16 + i),
            accumulated_gradient
        );
    }
}

// Template specializations.
__global__ void RasterizeGradKernel  (const RasterizeGradParams p) { RasterizeGradKernelTemplate<false>(p); }
__global__ void RasterizeGradKernelDb(const RasterizeGradParams p) { RasterizeGradKernelTemplate<true>(p); }

//------------------------------------------------------------------------
