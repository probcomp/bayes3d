// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "jax_interpolate.h"



//------------------------------------------------------------------------
// Kernel prototypes.

void InterpolateFwdKernel   (const InterpolateKernelParams p);
void InterpolateFwdKernelDa (const InterpolateKernelParams p);
void InterpolateGradKernel  (const InterpolateKernelParams p);
void InterpolateGradKernelDa(const InterpolateKernelParams p);

//------------------------------------------------------------------------
// Helper

static void set_diff_attrs(InterpolateKernelParams& p, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)
{
    if (diff_attrs_all)
    {
        p.numDiffAttr = p.numAttr;
        p.diff_attrs_all = 1;
    }
    else
    {
        NVDR_CHECK(diff_attrs_vec.size() <= IP_MAX_DIFF_ATTRS, "too many entries in diff_attrs list (increase IP_MAX_DIFF_ATTRS)");
        p.numDiffAttr = diff_attrs_vec.size();
        memcpy(p.diffAttrs, &diff_attrs_vec[0], diff_attrs_vec.size()*sizeof(int));
    }
}

//------------------------------------------------------------------------
// Forward op.

void _interpolate_fwd_da(cudaStream_t stream,
                        const float* attr, const float* rast, const int* tri,
                        const float* rast_db, bool diff_attrs_all,
                        std::vector<int>& diff_attrs_vec,
                        std::vector<int> attr_shape, std::vector<int> rast_shape,
                        std::vector<int> tri_shape,
                        float* out, float* out_da)

{
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    bool enable_da = (diff_attrs_all || !diff_attrs_vec.empty());;
    p.instance_mode = 1;  // assume instanced mode for JAX renderer.

    // Extract input dimensions; assume instanced mode.
    p.numVertices  = attr_shape[1];
    p.numAttr      = attr_shape[2];
    p.numTriangles = tri_shape[0];
    p.height       = rast_shape[1];
    p.width        = rast_shape[2];
    p.depth        = rast_shape[0];


    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enable_da)
        set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
    else
        p.numDiffAttr = 0;

    // Get input/output pointers.
    p.attr = attr;
    p.rast = rast;
    p.tri = tri;
    p.rastDB = enable_da ? rast_db : NULL;
    p.attrBC = (p.instance_mode && attr_shape[0] == 1) ? 1 : 0;

    p.out = out;
    p.outDA = enable_da ? out_da : NULL;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast   & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.outDA  &  7), "out_da output tensor not aligned to float2");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_FWD_MAX_KERNEL_BLOCK_WIDTH, IP_FWD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = enable_da ? (void*)InterpolateFwdKernelDa : (void*)InterpolateFwdKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}

void jax_interpolate_fwd(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {

    const DiffInterpolateCustomCallDescriptor &d =
        *UnpackDescriptor<DiffInterpolateCustomCallDescriptor>(opaque, opaque_len);

    const float *attr = reinterpret_cast<const float *> (buffers[0]);
    const float *rast_out = reinterpret_cast<const float *> (buffers[1]);
    const int *tri = reinterpret_cast<const int *> (buffers[2]);
    const float *rast_db = reinterpret_cast<const float *> (buffers[3]);
    const int *diff_attrs = reinterpret_cast<const int *> (buffers[4]);

    float *out = reinterpret_cast<float *> (buffers[5]);
    float *out_da = reinterpret_cast<float *> (buffers[6]);

    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<int> attr_shape;
    attr_shape.push_back(d.num_images);
    attr_shape.push_back(d.num_vertices);
    attr_shape.push_back(d.num_attributes);

    std::vector<int> rast_shape;
    rast_shape.push_back(d.rast_depth);
    rast_shape.push_back(d.rast_height);
    rast_shape.push_back(d.rast_width);

    std::vector<int> tri_shape;
    tri_shape.push_back(d.num_triangles);

    int num_diff_attrs = d.num_diff_attributes;
    bool diff_attrs_all = (num_diff_attrs == d.num_attributes);

    std::vector<int> diff_attrs_vec;

    if (diff_attrs_all){
        diff_attrs_vec.resize(0);
    }else{
        NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&diff_attrs_vec[0], diff_attrs, num_diff_attrs * sizeof(int), cudaMemcpyDeviceToHost));
    }

    cudaStreamSynchronize(stream);
    _interpolate_fwd_da(stream,
                    attr,
                    rast_out,
                    tri,
                    rast_db,
                    diff_attrs_all,
                    diff_attrs_vec,
                    attr_shape,
                    rast_shape,
                    tri_shape,
                    out,
                    out_da
                    );
    cudaStreamSynchronize(stream);
}

//------------------------------------------------------------------------
// Gradient op.

void _interpolate_grad_da(cudaStream_t stream,
                        const float* attr, const float* rast, const int* tri, const float* dy,
                        const float* rast_db, const float* dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec,
                        std::vector<int> attr_shape, std::vector<int> rast_shape, std::vector<int> tri_shape,
                        float* g_attr, float* g_rast, float* g_rast_db)
{
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    bool enable_da = (diff_attrs_all || !diff_attrs_vec.empty());
    p.instance_mode = 1;  // hardcoded

    NVDR_CHECK(enable_da, "ENABLE DA");

    // Depth of attributes.
    int attr_depth = p.instance_mode ? attr_shape[0] : 1;

    // Extract input dimensions; assume instanced mode.
    p.numVertices  = attr_shape[1];
    p.numAttr      = attr_shape[2];
    p.numTriangles = tri_shape[0];
    p.height       = rast_shape[1];
    p.width        = rast_shape[2];
    p.depth        = rast_shape[0];

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enable_da)
        set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
    else
        p.numDiffAttr = 0;

    // Get input pointers.
    p.attr = attr;
    p.rast = rast;
    p.tri = tri;
    p.dy = dy;
    p.rastDB = enable_da ? rast_db : NULL;
    p.dda = enable_da ? dda : NULL;
    p.attrBC = 0;

    // Allocate output tensors.
    p.gradAttr = g_attr;
    p.gradRaster = g_rast;
    p.gradRasterDB = enable_da ? g_rast_db : NULL;  // assuming enable_da = false

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.rast         & 15), "rast input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.rastDB       & 15), "rast_db input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dda          &  7), "dda input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.gradRaster   & 15), "grad_rast output tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.gradRasterDB & 15), "grad_rast_db output tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH, IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel.
    void* args[] = {&p};
    void* func = enable_da ? (void*)InterpolateGradKernelDa : (void*)InterpolateGradKernel;


    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}


void jax_interpolate_bwd(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {

    const DiffInterpolateCustomCallDescriptor &d =
        *UnpackDescriptor<DiffInterpolateCustomCallDescriptor>(opaque, opaque_len);

    const float *attr = reinterpret_cast<const float *> (buffers[0]);
    const float *rast_out = reinterpret_cast<const float *> (buffers[1]);
    const int *tri = reinterpret_cast<const int *> (buffers[2]);
    const float *dy = reinterpret_cast<const float *> (buffers[3]);
    const float *rast_db = reinterpret_cast<const float *> (buffers[4]);
    const float *dda = reinterpret_cast<const float *> (buffers[5]);
    const int *diff_attrs = reinterpret_cast<const int *> (buffers[6]);

    float *g_attr = reinterpret_cast<float *> (buffers[7]);
    float *g_rast = reinterpret_cast<float *> (buffers[8]);
    float* g_rast_db = reinterpret_cast<float *> (buffers[9]);
    cudaMemset(g_attr, 0, d.num_images*d.num_vertices*d.num_attributes*sizeof(float));

    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<int> attr_shape;
    attr_shape.push_back(d.num_images);
    attr_shape.push_back(d.num_vertices);
    attr_shape.push_back(d.num_attributes);

    std::vector<int> rast_shape;
    rast_shape.push_back(d.rast_depth);
    rast_shape.push_back(d.rast_height);
    rast_shape.push_back(d.rast_width);

    std::vector<int> tri_shape;
    tri_shape.push_back(d.num_triangles);

    int num_diff_attrs = d.num_diff_attributes;
    bool diff_attrs_all = (num_diff_attrs == d.num_attributes);

    std::vector<int> diff_attrs_vec;

    if (diff_attrs_all){
        diff_attrs_vec.resize(0);
    }else{
        NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&diff_attrs_vec[0], diff_attrs, num_diff_attrs * sizeof(int), cudaMemcpyDeviceToHost));
    }

    cudaStreamSynchronize(stream);
    _interpolate_grad_da(stream,
                    attr,
                    rast_out,
                    tri,
                    dy,
                    rast_db, //
                    dda, //
                    diff_attrs_all, //
                    diff_attrs_vec, //
                    attr_shape,
                    rast_shape,
                    tri_shape,
                    g_attr,
                    g_rast,
                    g_rast_db //
                    );
    cudaStreamSynchronize(stream);
}

// //------------------------------------------------------------------------
