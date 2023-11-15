// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "jax_binding_ops.h"
#include "jax_interpolate.h"
#include "../common/interpolate.h"
#include "../common/common.h"


//------------------------------------------------------------------------
// Kernel prototypes.

void InterpolateFwdKernel   (const InterpolateKernelParams p);
void InterpolateFwdKernelDa (const InterpolateKernelParams p);
void InterpolateGradKernel  (const InterpolateKernelParams p);
void InterpolateGradKernelDa(const InterpolateKernelParams p);

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
    bool enable_da = false;
    p.instance_mode = 1;  // assume instanced mode for JAX renderer.

    NVDR_CHECK(enable_da == false, "No support for attribute grads.");

    // Extract input dimensions; assume instanced mode.
    p.numVertices  = attr_shape[1];
    p.numAttr      = attr_shape[2];
    p.numTriangles = tri_shape[0];
    p.height       = rast_shape[1];
    p.width        = rast_shape[2];
    p.depth        = rast_shape[0];

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
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

void _interpolate_fwd(cudaStream_t stream, const float* attr, const float* rast, const int* tri, 
                    std::vector<int> attr_shape, std::vector<int> rast_shape, std::vector<int> tri_shape, 
                    float* out, float* out_da)
{
    std::vector<int> empty_vec;
    const float* empty_tensor;
    _interpolate_fwd_da(stream, attr, rast, tri, empty_tensor, false, 
                        empty_vec, attr_shape, rast_shape, tri_shape, 
                        out, out_da);
}

void jax_interpolate_fwd(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {

    const DiffInterpolateCustomCallDescriptor &d = 
        *UnpackDescriptor<DiffInterpolateCustomCallDescriptor>(opaque, opaque_len);

    const float *attr = reinterpret_cast<const float *> (buffers[0]);
    const float *rast_out = reinterpret_cast<const float *> (buffers[1]);
    const int *tri = reinterpret_cast<const int *> (buffers[2]);

    float *out = reinterpret_cast<float *> (buffers[3]);
    float *_out_da_dummy = reinterpret_cast<float *> (buffers[4]);  // because no attribute grad, this output is meaningless
    
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

    cudaStreamSynchronize(stream);
    _interpolate_fwd(stream,
                    attr, 
                    rast_out, 
                    tri, 
                    attr_shape, 
                    rast_shape, 
                    tri_shape, 
                    out,
                    _out_da_dummy
                    );
    cudaStreamSynchronize(stream);
}

//------------------------------------------------------------------------
// Gradient op.

void _interpolate_grad_da(cudaStream_t stream, 
                        const float* attr, const float* rast, const int* tri, const float* dy,
                        const float* rast_db, const float* dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec,
                        std::vector<int> attr_shape, std::vector<int> rast_shape, std::vector<int> tri_shape, 
                        float* g_attr, float* g_rast)
{
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    bool enable_da = false;
    p.instance_mode = 1;  // hardcoded

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
    p.numDiffAttr = 0;

    // Get input pointers.
    p.attr = attr;
    p.rast = rast;
    p.tri = tri;
    p.dy = dy;
    p.rastDB = NULL;
    p.dda = NULL;
    p.attrBC = 0;

    // Allocate output tensors.
    p.gradAttr = g_attr;
    p.gradRaster = g_rast;
    p.gradRasterDB = NULL;  // assuming enable_da = false

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

// Version without derivatives.
void _interpolate_grad(cudaStream_t stream, const float* attr, const float* rast, const int* tri, const float* dy,
                    std::vector<int> attr_shape, std::vector<int> rast_shape, std::vector<int> tri_shape, 
                    float* g_attr, float* g_rast)
{

    std::vector<int> empty_vec;
    const float* empty_tensor;
    _interpolate_grad_da(stream, 
                        attr, rast, tri, dy, 
                        empty_tensor, empty_tensor, false, empty_vec,
                        attr_shape, rast_shape, tri_shape,
                        g_attr, g_rast);
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

    float *g_attr = reinterpret_cast<float *> (buffers[4]);
    float *g_rast = reinterpret_cast<float *> (buffers[5]);  
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

    cudaStreamSynchronize(stream);
    _interpolate_grad(stream,
                    attr,
                    rast_out, 
                    tri, 
                    dy,
                    attr_shape, 
                    rast_shape, 
                    tri_shape, 
                    g_attr,
                    g_rast
                    );
    cudaStreamSynchronize(stream);
}

// //------------------------------------------------------------------------
