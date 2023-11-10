// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include "interpolate_gl_jax.h"
#include "../common/common.h"
#include "../common/interpolate.h"

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

std::tuple<torch::Tensor, torch::Tensor> _interpolate_fwd_da(cudaStream_t stream, 
                                                        const float* attr, const float* rast, const int* tri, 
                                                        const float* rast_db, bool diff_attrs_all, 
                                                        std::vector<int>& diff_attrs_vec)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(attr));
    InterpolateKernelParams p = {}; // Initialize all fields to zero.
    bool enable_da = (rast_db.defined()) && (diff_attrs_all || !diff_attrs_vec.empty());
    p.instance_mode = 1;  // assume instanced mode for JAX renderer.

    NVDR_CHECK(enable_da == False, "No support for attribute grads.");

    // Check inputs.
    if (enable_da)
    {
        NVDR_CHECK_DEVICE(attr, rast, tri, rast_db);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri, rast_db);
        NVDR_CHECK_F32(attr, rast, rast_db);
        NVDR_CHECK_I32(tri);
    }
    else
    {
        NVDR_CHECK_DEVICE(attr, rast, tri);
        NVDR_CHECK_CONTIGUOUS(attr, rast, tri);
        NVDR_CHECK_F32(attr, rast);
        NVDR_CHECK_I32(tri);
    }

    // Sanity checks.
    NVDR_CHECK(rast.sizes().size() == 4 && rast.size(0) > 0 && rast.size(1) > 0 && rast.size(2) > 0 && rast.size(3) == 4, "rast must have shape[>0, >0, >0, 4]");
    NVDR_CHECK( tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");
    NVDR_CHECK((attr.sizes().size() == 2 || attr.sizes().size() == 3) && attr.size(0) > 0 && attr.size(1) > 0 && (attr.sizes().size() == 2 || attr.size(2) > 0), "attr must have shape [>0, >0, >0] or [>0, >0]");
    if (p.instance_mode)
        NVDR_CHECK(attr.size(0) == rast.size(0) || attr.size(0) == 1, "minibatch size mismatch between inputs rast, attr");
    if (enable_da)
    {
        NVDR_CHECK(rast_db.sizes().size() == 4 && rast_db.size(0) > 0 && rast_db.size(1) > 0 && rast_db.size(2) > 0 && rast_db.size(3) == 4, "rast_db must have shape[>0, >0, >0, 4]");
        NVDR_CHECK(rast_db.size(1) == rast.size(1) && rast_db.size(2) == rast.size(2), "spatial size mismatch between inputs rast and rast_db");
        NVDR_CHECK(rast_db.size(0) == rast.size(0), "minibatch size mismatch between inputs rast, rast_db");
    }

    // Extract input dimensions.
    p.numVertices  = attr.size(p.instance_mode ? 1 : 0);
    p.numAttr      = attr.size(p.instance_mode ? 2 : 1);
    p.numTriangles = tri.size(0);
    p.height       = rast.size(1);
    p.width        = rast.size(2);
    p.depth        = rast.size(0);

    // Set attribute pixel differential info if enabled, otherwise leave as zero.
    if (enable_da)
        set_diff_attrs(p, diff_attrs_all, diff_attrs_vec);
    else
        p.numDiffAttr = 0;

    // Get input pointers.
    p.attr = attr.data_ptr<float>();
    p.rast = rast.data_ptr<float>();
    p.tri = tri.data_ptr<int>();
    p.rastDB = enable_da ? rast_db.data_ptr<float>() : NULL;
    p.attrBC = (p.instance_mode && attr.size(0) == 1) ? 1 : 0;

    // Allocate output tensors.
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({p.depth, p.height, p.width, p.numAttr}, opts);
    torch::Tensor out_da = torch::empty({p.depth, p.height, p.width, p.numDiffAttr * 2}, opts);

    p.out = out.data_ptr<float>();
    p.outDA = enable_da ? out_da.data_ptr<float>() : NULL;

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

    // Return results.
    return std::tuple<torch::Tensor, torch::Tensor>(out, out_da);
}

std::tuple<torch::Tensor, torch::Tensor> _interpolate_fwd(cudaStream_t stream, torch::Tensor attr, torch::Tensor rast, torch::Tensor tri)
{
    std::vector<int> empty_vec;
    const float* empty_tensor;
    return interpolate_fwd_da(stream, attr, rast, tri, empty_tensor, false, empty_vec);
}

void jax_interpolate_fwd_gl(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {

    const DiffRasterizeCustomCallDescriptor &d = 
        *UnpackDescriptor<DiffRasterizeCustomCallDescriptor>(opaque, opaque_len);
    RasterizeGLStateWrapper& stateWrapper = *d.gl_state_wrapper;

    const float *posw = reinterpret_cast<const float *> (buffers[0]);
    const int *rast_out = reinterpret_cast<const int *> (buffers[1]);
    const int *pos_idx = reinterpret_cast<const int *> (buffers[2]);

    float *out = reinterpret_cast<float *> (buffers[3]);
    float *_out_dummy = reinterpret_cast<float *> (buffers[4]);  // because no attribute grad, this output is meaningless
    
    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    cudaStreamSynchronize(stream);
    _interpolate_fwd(stream,
                    posw, 
                    rast_out, 
                    pos_idx
                    );
    cudaStreamSynchronize(stream);
}

//------------------------------------------------------------------------
// Gradient op.

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolate_grad_da(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy, torch::Tensor rast_db, torch::Tensor dda, bool diff_attrs_all, std::vector<int>& diff_attrs_vec)

// Version without derivatives.
// std::tuple<torch::Tensor, torch::Tensor> interpolate_grad(torch::Tensor attr, torch::Tensor rast, torch::Tensor tri, torch::Tensor dy)

//------------------------------------------------------------------------

//--------------------------------------------------------
// Registrations
//--------------------------------------------------------

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["jax_interpolate_fwd_gl"] = EncapsulateFunction(jax_interpolate_fwd_gl);
  return dict;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Ops.
    m.def("registrations", &Registrations, "custom call registrations");
    m.def("build_diff_interpolate_descriptor",
            [](InterpolateStateWrapper& stateWrapper,
            std::vector<int> images_vertices_triangles) {
            DiffInterpolateCustomCallDescriptor d;
            d.num_images = images_vertices_triangles[0];
            d.num_vertices = images_vertices_triangles[1];
            d.num_triangles = images_vertices_triangles[2];
            return PackDescriptor(d);
        });
        // opaque = dr._get_plugin(gl=True).build_rasterize_descriptor(r.renderer_env.cpp_wrapper,
        //                                                             [num_images, num_vertices, num_triangles])
}
