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

    const float *posw = reinterpret_cast<const float *> (buffers[0]);
    const float *rast_out = reinterpret_cast<const float *> (buffers[1]);
    const int *pos_idx = reinterpret_cast<const int *> (buffers[2]);

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
                    posw, 
                    rast_out, 
                    pos_idx, 
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

// void _interpolate_grad_da(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy, torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)
// {
//     // const at::cuda::OptionalCUDAGuard device_guard(device_of(attr));
//     // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     InterpolateKernelParams p = {}; // Initialize all fields to zero.
//     bool enable_da = (rast_db.defined()) && (diff_attrs_all || !diff_attrs_vec.empty());
//     p.instance_mode = (attr.sizes().size() > 2) ? 1 : 0;

//     // Check inputs.
//     if (enable_da)
//     {
//         NVDR_CHECK_DEVICE(attr, rast, tri, dy, rast_db, dda);
//         NVDR_CHECK_CONTIGUOUS(attr, rast, tri, rast_db);
//         NVDR_CHECK_F32(attr, rast, dy, rast_db, dda);
//         NVDR_CHECK_I32(tri);
//     }
//     else
//     {
//         NVDR_CHECK_DEVICE(attr, rast, tri, dy);
//         NVDR_CHECK_CONTIGUOUS(attr, rast, tri);
//         NVDR_CHECK_F32(attr, rast, dy);
//         NVDR_CHECK_I32(tri);
//     }

//     // Depth of attributes.
//     int attr_depth = p.instance_mode ? (attr.sizes().size() > 1 ? attr.size(0) : 0) : 1;

//     // Sanity checks.
//     NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "rast must have shape[>0, >0, >0, 4]");
//     NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
//     NVDR_CHECK((attr.sizes().size() == 2 || attr.sizes().size() == 3) && attr.size(0) > 0 && attr.size(1) > 0 && (attr.sizes().size() == 2 || attr.size(2) > 0), "attr must have shape [>0, >0, >0] or [>0, >0]");
//     NVDR_CHECK(dy.sizes().size() == 4 && dy.size(0) > 0 && dy.size(1) == rast.size(1) && dy.size(2) == rast.size(2) && dy.size(3) > 0, "dy must have shape [>0, height, width, >0]");
//     NVDR_CHECK(dy.size(3) == attr.size(attr.sizes().size() - 1), "argument count mismatch between inputs dy, attr");
//     NVDR_CHECK((attr_depth == rast.size(0) || attr_depth == 1) && dy.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, dy, attr");
//     if (enable_da)
//     {
//         NVDR_CHECK(dda.sizes().size() == 4 && dda.size(0) > 0 && dda.size(1) == rast.size(1) && dda.size(2) == rast.size(2), "dda must have shape [>0, height, width, ?]");
//         NVDR_CHECK(dda.size(0) == rast.size(0), "minibatch size mismatch between rast, dda");
//         NVDR_CHECK(rast_db.sizes().size() == 4 && rast_db.size(0) > 0 && rast_db.size(1) > 0 && rast_db.size(2) > 0 && rast_db.size(3) == 4, "rast_db must have shape[>0, >0, >0, 4]");
//         NVDR_CHECK(rast_db.size(1) == rast.size(1) && rast_db.size(2) == rast.size(2), "spatial size mismatch between inputs rast and rast_db");
//         NVDR_CHECK(rast_db.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, rast_db");
//     }

//     // Extract input dimensions.
//     p.numVertices  = attr.size(p.instance_mode ? 1 : 0);
//     p.numAttr      = attr.size(p.instance_mode ? 2 : 1);
//     p.numTriangles = tri.size(0);
//     p.height       = rast.size(1);
//     p.width        = rast.size(2);
//     p.depth        = rast.size(0);

//     // Ensure gradients are contiguous.
//     torch::Tensor dy_ = dy.contiguous();
//     torch::Tensor dda_;
//     if (enable_da)
//         dda_ = dda.contiguous();

//     // Set attribute pixel differential info if enabled, otherwise leave as zero.
//     if (enable_da)
//         set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
//     else
//         p.numDiffAttr = 0;

//     // Get input pointers.
//     p.attr = attr.data_ptr<float>();
//     p.rast = rast.data_ptr<float>();
//     p.tri = tri.data_ptr<int>();
//     p.dy = dy_.data_ptr<float>();
//     p.rastDB = enable_da ? rast_db.data_ptr<float>() : NULL;
//     p.dda = enable_da ? dda_.data_ptr<float>() : NULL;
//     p.attrBC = (p.instance_mode && attr_depth < p.depth) ? 1 : 0;

//     // Allocate output tensors.
//     torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//     torch::Tensor gradAttr = torch::zeros_like(attr);
//     torch::Tensor gradRaster = torch::empty_like(rast);
//     torch::Tensor gradRasterDB;
//     if (enable_da)
//         gradRasterDB = torch::empty_like(rast_db);

//     p.gradAttr = gradAttr.data_ptr<float>();
//     p.gradRaster = gradRaster.data_ptr<float>();
//     p.gradRasterDB = enable_da ? gradRasterDB.data_ptr<float>() : NULL;

//     // Verify that buffers are aligned to allow float2/float4 operations.
//     NVDR_CHECK(!((uintptr_t)p.rast         & 15), "rast input tensor not aligned to float4");
//     NVDR_CHECK(!((uintptr_t)p.rastDB       & 15), "rast_db input tensor not aligned to float4");
//     NVDR_CHECK(!((uintptr_t)p.dda          &  7), "dda input tensor not aligned to float2");
//     NVDR_CHECK(!((uintptr_t)p.gradRaster   & 15), "grad_rast output tensor not aligned to float4");
//     NVDR_CHECK(!((uintptr_t)p.gradRasterDB & 15), "grad_rast_db output tensor not aligned to float4");

//     // Choose launch parameters.
//     dim3 blockSize = getLaunchBlockSize(IP_GRAD_MAX_KERNEL_BLOCK_WIDTH, IP_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
//     dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

//     // Launch CUDA kernel.
//     void* args[] = {&p};
//     void* func = enable_da ? (void*)InterpolateGradKernelDa : (void*)InterpolateGradKernel;
//     NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

//     // // Return results.
//     // return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>(gradAttr, gradRaster, gradRasterDB);
// }

// // Version without derivatives.
// void _interpolate_grad(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy)
// {
//     std::vector<int> empty_vec;
//     torch::Tensor empty_tensor;
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> result = _interpolate_grad_da(attr, rast, tri, dy, empty_tensor, empty_tensor, false, empty_vec);
//     return std::tuple<torch::Tensor, torch::Tensor>(std::get<0>(result), std::get<1>(result));
// }

// void jax_interpolate_bwd(cudaStream_t stream,
//                           void **buffers,
//                           const char *opaque, std::size_t opaque_len) {

//     const DiffInterpolateCustomCallDescriptor &d = 
//         *UnpackDescriptor<DiffInterpolateCustomCallDescriptor>(opaque, opaque_len);

//     const float *posw = reinterpret_cast<const float *> (buffers[0]);
//     const float *rast_out = reinterpret_cast<const float *> (buffers[1]);
//     const int *pos_idx = reinterpret_cast<const int *> (buffers[2]);

//     float *out = reinterpret_cast<float *> (buffers[3]);
//     float *_out_da_dummy = reinterpret_cast<float *> (buffers[4]);  // because no attribute grad, this output is meaningless
    
//     auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

//     std::vector<int> attr_shape;
//     attr_shape.push_back(d.num_images);
//     attr_shape.push_back(d.num_vertices);
//     attr_shape.push_back(d.num_attributes);

//     std::vector<int> rast_shape;
//     rast_shape.push_back(d.rast_depth);
//     rast_shape.push_back(d.rast_height);
//     rast_shape.push_back(d.rast_width);

//     std::vector<int> tri_shape;
//     tri_shape.push_back(d.num_triangles);

//     cudaStreamSynchronize(stream);
//     _interpolate_grad(stream,
//                     posw, 
//                     rast_out, 
//                     pos_idx, 
//                     attr_shape, 
//                     rast_shape, 
//                     tri_shape, 
//                     out,
//                     _out_da_dummy
//                     );
//     cudaStreamSynchronize(stream);
// }

//------------------------------------------------------------------------
