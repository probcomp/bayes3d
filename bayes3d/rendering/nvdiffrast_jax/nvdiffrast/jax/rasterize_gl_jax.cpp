// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include "rasterize_gl_jax.h"
#include "../common/common.h"
#include <tuple>

//------------------------------------------------------------------------
// Python GL state wrapper methods.

RasterizeGLStateWrapper::RasterizeGLStateWrapper(bool enableDB, bool automatic_, int cudaDeviceIdx_)
{
    pState = new RasterizeGLState();
    automatic = automatic_;
    cudaDeviceIdx = cudaDeviceIdx_;
    memset(pState, 0, sizeof(RasterizeGLState));
    pState->enableDB = enableDB ? 1 : 0;
    rasterizeInitGLContext(NVDR_CTX_PARAMS, *pState, cudaDeviceIdx_);
    releaseGLContext();
}

RasterizeGLStateWrapper::~RasterizeGLStateWrapper(void)
{
    setGLContext(pState->glctx);
    rasterizeReleaseBuffers(NVDR_CTX_PARAMS, *pState);
    releaseGLContext();
    destroyGLContext(pState->glctx);
    delete pState;
}

void RasterizeGLStateWrapper::setContext(void)
{
    setGLContext(pState->glctx);
}

void RasterizeGLStateWrapper::releaseContext(void)
{
    releaseGLContext();
}

//------------------------------------------------------------------------
// Forward op (OpenGL).

// void _rasterize_fwd_gl(cudaStream_t stream, RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, std::tuple<int, int> resolution, torch::Tensor ranges, int peeling_idx)
void _rasterize_fwd_gl(cudaStream_t stream, RasterizeGLStateWrapper& stateWrapper, 
                        const float* pos, const int* tri, 
                        std::vector<int> dims,
                        std::vector<int> resolution, 
                        float* out)
                        // float* out_db)
{ 
    // const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pos));
    RasterizeGLState& s = *stateWrapper.pState;

    // // Check inputs.
    // NVDR_CHECK_DEVICE(pos, tri);
    // NVDR_CHECK_F32(pos);

    // // Check that GL context was created for the correct GPU.
    // NVDR_CHECK(pos.get_device() == stateWrapper.cudaDeviceIdx, "GL context must must reside on the same device as input tensors");

    // Determine number of outputs
    int num_outputs = s.enableDB ? 2 : 1;

    // Get output shape.
    int height = resolution[0];
    int width  = resolution[1];
    int depth = dims[0];
    // int depth  = instance_mode ? pos.size(0) : ranges.size(0);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0];");

    // Get position and triangle buffer sizes in int32/float32.
    int posCount = 4 * dims[0] * dims[1];
    int triCount = 3 * dims[2];

    // Set the GL context unless manual context.
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // Resize all buffers.
    bool changes = false;
    rasterizeResizeBuffers(NVDR_CTX_PARAMS, s, changes, posCount, triCount, width, height, depth);
    if (changes)
    {
#ifdef _WIN32
        // Workaround for occasional blank first frame on Windows.
        releaseGLContext();
        setGLContext(s.glctx);
#endif
    }

    // // Copy input data to GL and render.
    // const float* posPtr = pos.data_ptr<float>();
    // const int32_t* rangesPtr = instance_mode ? 0 : ranges.data_ptr<int32_t>(); // This is in CPU memory.
    // const int32_t* triPtr = tri.data_ptr<int32_t>();
    // int vtxPerInstance = instance_mode ? pos.size(1) : 0;
    // rasterizeRender(NVDR_CTX_PARAMS, s, stream, posPtr, posCount, vtxPerInstance, triPtr, triCount, rangesPtr, width, height, depth, peeling_idx);
    int peeling_idx = -1;
    const float* posPtr = pos;
    const int32_t* rangesPtr = 0; // This is in CPU memory.
    const int32_t* triPtr = tri;
    int vtxPerInstance = dims[1];
    rasterizeRender(NVDR_CTX_PARAMS, s, stream, posPtr, posCount, vtxPerInstance, triPtr, triCount, rangesPtr, width, height, depth, peeling_idx);

    // // Allocate output tensors.
    // torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // torch::Tensor out = torch::empty({depth, height, width, 4}, opts);
    // torch::Tensor out_db = torch::empty({depth, height, width, s.enableDB ? 4 : 0}, opts);
    // float* outputPtr[2];
    // outputPtr[0] = out.data_ptr<float>();
    // outputPtr[1] = s.enableDB ? out_db.data_ptr<float>() : NULL;
    float* outputPtr[2];
    outputPtr[0] = out;
    outputPtr[1] = NULL;// s.enableDB ? out_db : NULL;

    // Copy rasterized results into CUDA buffers.
    rasterizeCopyResults(NVDR_CTX_PARAMS, s, stream, outputPtr, width, height, depth);

    // Done. Release GL context and return.
    if (stateWrapper.automatic)
        releaseGLContext();
}

void jax_rasterize_fwd_gl(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {

    const DiffRasterizeCustomCallDescriptor &d = 
        *UnpackDescriptor<DiffRasterizeCustomCallDescriptor>(opaque, opaque_len);
    RasterizeGLStateWrapper& stateWrapper = *d.gl_state_wrapper;

    void *pos = buffers[0];
    void *tri = buffers[1];
    void *_resolution = buffers[3];

    void *out = buffers[4];
    // void *out_db = buffers[5];   // TODO (see jax documentation on modifications for multiple outputs)
    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<int> resolution;
    resolution.resize(2);
    std::vector<int> pos_dim;
    pos_dim.resize(3);

    cudaStreamSynchronize(stream);
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&resolution[0], _resolution, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    pos_dim[0] = d.num_images;
    pos_dim[1] = d.num_vertices;
    pos_dim[2] = d.num_triangles;

    _rasterize_fwd_gl(stream,
                      stateWrapper,
                      reinterpret_cast<const float *>(pos),
                      reinterpret_cast<const int *>(tri),
                      pos_dim,
                      resolution, 
                      reinterpret_cast<float *>(out) 
                    //   reinterpret_cast<float *>(out_db)  
                      );
    cudaStreamSynchronize(stream);
}

//--------------------------------------------------------
// Registrations
//--------------------------------------------------------

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["jax_rasterize_fwd_gl"] = EncapsulateFunction(jax_rasterize_fwd_gl);
  return dict;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeGLStateWrapper>(m, "RasterizeGLStateWrapper").def(pybind11::init<bool, bool, int>())
        .def("set_context",     &RasterizeGLStateWrapper::setContext)
        .def("release_context", &RasterizeGLStateWrapper::releaseContext);

    // Ops.
    m.def("registrations", &Registrations, "custom call registrations");
    m.def("build_diff_rasterize_descriptor",
            [](RasterizeGLStateWrapper& stateWrapper,
            std::vector<int> images_vertices_triangles) {
            DiffRasterizeCustomCallDescriptor d;
            d.gl_state_wrapper = &stateWrapper;
            d.num_images = images_vertices_triangles[0];
            d.num_vertices = images_vertices_triangles[1];
            d.num_triangles = images_vertices_triangles[2];
            return PackDescriptor(d);
        });
        // opaque = dr._get_plugin(gl=True).build_rasterize_descriptor(r.renderer_env.cpp_wrapper,
        //                                                             [num_images, num_vertices, num_triangles])
}

//------------------------------------------------------------------------





