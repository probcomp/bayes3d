// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "torch_common.inl"
#include "torch_types.h"
#include "../common/common.h"
#include "../common/rasterize.h"
#include "jax_rasterize_gl.h"
#include "jax_binding_ops.h"
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
                        float* out,
                        float* out_db)
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
    int peeling_idx = -1;
    const float* posPtr = pos;
    const int32_t* rangesPtr = 0; // This is in CPU memory.
    const int32_t* triPtr = tri;
    int vtxPerInstance = dims[1];
    rasterizeRender(NVDR_CTX_PARAMS, s, stream, posPtr, posCount, vtxPerInstance, triPtr, triCount, rangesPtr, width, height, depth, peeling_idx);

    // Allocate output tensors.
    float* outputPtr[2];
    outputPtr[0] = out;
    outputPtr[1] = s.enableDB ? out_db : NULL;

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

    const float *pos = reinterpret_cast<const float *> (buffers[0]);
    const int *tri = reinterpret_cast<const int *> (buffers[1]);
    const int *_resolution = reinterpret_cast<const int *> (buffers[2]);

    float *out = reinterpret_cast<float *> (buffers[3]);
    float *out_db = reinterpret_cast<float *> (buffers[4]);
    
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
                      pos,
                      tri,
                      pos_dim,
                      resolution, 
                      out,
                      out_db
                      );
    cudaStreamSynchronize(stream);
}

//========================================================================
// Gradient op.

//------------------------------------------------------------------------
// Kernel prototypes.

void RasterizeGradKernel(const RasterizeGradParams p);
void RasterizeGradKernelDb(const RasterizeGradParams p);
//------------------------------------------------------------------------

void _rasterize_grad_db(cudaStream_t stream,
                        const float* pos, const int* tri, const float* rast_out, 
                        const float* dy, const float* ddb, 
                        std::vector<int> pos_shape,  
                        std::vector<int> tri_shape,  
                        std::vector<int> rast_out_shape,
                        float* grad)
{
    RasterizeGradParams p;
    bool enable_db = true;

    // Determine instance mode.
    p.instance_mode = 1;
    NVDR_CHECK(p.instance_mode == 1, "Should be in instance mode; check input sizes");

    // Shape is taken from the rasterizer output tensor.
    p.depth  = rast_out_shape[0];
    p.height = rast_out_shape[1];
    p.width  = rast_out_shape[2];
    NVDR_CHECK(p.depth > 0 && p.height > 0 && p.width > 0, "resolution must be [>0, >0, >0]");

    // Populate parameters.
    p.numTriangles = tri_shape[0];
    p.numVertices = p.instance_mode ? pos_shape[1] : pos_shape[0];
    p.pos = pos;
    p.tri = tri;
    p.out = rast_out;
    p.dy  = dy;
    p.ddb = enable_db ? ddb : NULL;

    // Set up pixel position to clip space x, y transform.
    p.xs = 2.f / (float)p.width;
    p.xo = 1.f / (float)p.width - 1.f;
    p.ys = 2.f / (float)p.height;
    p.yo = 1.f / (float)p.height - 1.f;

    // Output tensor for position gradients.
    p.grad = grad;

    // Verify that buffers are aligned to allow float2/float4 operations.
    NVDR_CHECK(!((uintptr_t)p.pos & 15), "pos input tensor not aligned to float4");
    NVDR_CHECK(!((uintptr_t)p.dy  &  7), "dy input tensor not aligned to float2");
    NVDR_CHECK(!((uintptr_t)p.ddb & 15), "ddb input tensor not aligned to float4");

    // Choose launch parameters.
    dim3 blockSize = getLaunchBlockSize(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH, RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // Launch CUDA kernel to populate gradient values.
    void* args[] = {&p};
    void* func = enable_db ? (void*)RasterizeGradKernelDb : (void*)RasterizeGradKernel;
    NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));
}

void jax_rasterize_bwd(cudaStream_t stream,
                        void **buffers,
                        const char *opaque, std::size_t opaque_len) {

    const DiffRasterizeBwdCustomCallDescriptor &d = 
        *UnpackDescriptor<DiffRasterizeBwdCustomCallDescriptor>(opaque, opaque_len);

    const float *pos = reinterpret_cast<const float *> (buffers[0]);
    const int *tri = reinterpret_cast<const int *> (buffers[1]);
    const float *rast_out = reinterpret_cast<const float *> (buffers[2]);
    const float *dy = reinterpret_cast<const float *> (buffers[3]);
    const float *ddb = reinterpret_cast<const float *> (buffers[4]);

    float *grad = reinterpret_cast<float *> (buffers[5]);  // output
    cudaMemset(grad, 0, d.num_images*d.num_vertices*4*sizeof(float));
    
    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<int> pos_shape;
    pos_shape.resize(2);
    std::vector<int> tri_shape;
    tri_shape.resize(1);
    std::vector<int> rast_out_shape;
    rast_out_shape.resize(3);

    pos_shape[0] = d.num_images;
    pos_shape[1] = d.num_vertices;
    tri_shape[0] = d.num_triangles;
    rast_out_shape[0] = d.rast_depth;
    rast_out_shape[1] = d.rast_height;
    rast_out_shape[2] = d.rast_width;

    cudaStreamSynchronize(stream);
    _rasterize_grad_db(stream,
                      pos,
                      tri,
                      rast_out, 
                      dy, 
                      ddb,
                      pos_shape,
                      tri_shape, 
                      rast_out_shape,
                      grad
                      );
    cudaStreamSynchronize(stream);
}
