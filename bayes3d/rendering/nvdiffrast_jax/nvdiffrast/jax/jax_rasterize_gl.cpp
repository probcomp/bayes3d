// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "jax_rasterize_gl.h"
#include <tuple>

//------------------------------------------------------------------------
// Forward op (OpenGL).
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



void jax_rasterize_fwd_gl(cudaStream_t stream,
                          void **buffers,
                          const char *opaque, std::size_t opaque_len) {
    const DiffRasterizeCustomCallDescriptor &d =
        *UnpackDescriptor<DiffRasterizeCustomCallDescriptor>(opaque, opaque_len);
    RasterizeGLStateWrapper& stateWrapper = *d.gl_state_wrapper;

    const float *pose = reinterpret_cast<const float *> (buffers[0]);
    const float *pos = reinterpret_cast<const float *> (buffers[1]);
    const int *tri = reinterpret_cast<const int *> (buffers[2]);
    const int *_ranges = reinterpret_cast<const int *> (buffers[3]);
    const float *projectionMatrix = reinterpret_cast<const float *> (buffers[4]);
    const int *_resolution = reinterpret_cast<const int *> (buffers[5]);

    float *out = reinterpret_cast<float *> (buffers[6]);
    float *out_db = reinterpret_cast<float *> (buffers[7]);

    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    std::vector<int> resolution;
    resolution.resize(2);
    int ranges[2*d.num_objects];

    // std::cout << "num_images: " << d.num_images << std::endl;
    // std::cout << "num_objects: " << d.num_objects << std::endl;
    // std::cout << "num_vertices: " << d.num_vertices << std::endl;
    // std::cout << "num_triangles: " << d.num_triangles << std::endl;

    cudaStreamSynchronize(stream);
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&resolution[0], _resolution, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&ranges[0], _ranges, 2 * d.num_objects * sizeof(int), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(stream);

    // Allocate output tensors.
    cudaStreamSynchronize(stream);

    cudaStreamSynchronize(stream);

    // const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pos));
    RasterizeGLState& s = *stateWrapper.pState;

    // Determine number of outputs
    int num_outputs = s.enableDB ? 2 : 1;

    // Get output shape.
    int height = resolution[0];
    int width  = resolution[1];
    int depth = d.num_images;
    int max_parallel_images = 1024;
    // int depth  = instance_mode ? pos.size(0) : ranges.size(0);
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0];");

    // Get position and triangle buffer sizes in int32/float32.
    int posCount = 4 * d.num_vertices;
    int triCount = 3 * d.num_triangles;

    // Set the GL context unless manual context.
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // Resize all buffers.
    bool changes = false;
    cudaStreamSynchronize(stream);

    rasterizeResizeBuffers(NVDR_CTX_PARAMS, s, changes, posCount, triCount, width, height, depth);
    cudaStreamSynchronize(stream);

    if (changes)
    {
#ifdef _WIN32
        // Workaround for occasional blank first frame on Windows.
        releaseGLContext();
        setGLContext(s.glctx);
#endif
    }

    // Allocate output tensors.
    float* outputPtr[2];
    outputPtr[0] = out;
    outputPtr[1] = s.enableDB ? out_db : NULL;
    cudaMemset(out, 0, d.num_images*width*height*4*sizeof(float));
    cudaMemset(out_db, 0, d.num_images*width*height*4*sizeof(float));


    cudaStreamSynchronize(stream);
    std::vector<float> projMatrix;
    projMatrix.resize(16);
    NVDR_CHECK_CUDA_ERROR(cudaMemcpy(&projMatrix[0], projectionMatrix, 16 * sizeof(int), cudaMemcpyDeviceToHost));
    cudaStreamSynchronize(stream);



    // for(int i = 0; i < 16; i++) {
    //     std::cout << firstPose[i] << " ";
    // }
    // std::cout << std::endl;

    // // Copy input data to GL and render.
    int peeling_idx = -1;
    const float* posePtr = pose;
    const float* posPtr = pos;
    const int32_t* rangesPtr = ranges; // This is in CPU memory.
    const int32_t* triPtr = tri;
    cudaStreamSynchronize(stream);
    rasterizeRender(NVDR_CTX_PARAMS, s, stream, outputPtr, projMatrix, posePtr, posPtr, posCount, d.num_vertices, triPtr, triCount, rangesPtr, d.num_objects, width, height, depth, peeling_idx);
    cudaStreamSynchronize(stream);


    // // Copy rasterized results into CUDA buffers.
    // cudaStreamSynchronize(stream);
    // rasterizeCopyResults(NVDR_CTX_PARAMS, s, stream, outputPtr, width, height, depth);
    // cudaStreamSynchronize(stream);

    // Done. Release GL context and return.
    if (stateWrapper.automatic)
        releaseGLContext();

    cudaStreamSynchronize(stream);
}

//========================================================================
// Gradient op.

//------------------------------------------------------------------------
// Kernel prototypes.

void RasterizeGradKernel(const RasterizeGradParams p);
void RasterizeGradKernelDb(const RasterizeGradParams p);
//------------------------------------------------------------------------


void jax_rasterize_bwd(cudaStream_t stream,
                        void **buffers,
                        const char *opaque, std::size_t opaque_len) {

    // const DiffRasterizeBwdCustomCallDescriptor &d =
    //     *UnpackDescriptor<DiffRasterizeBwdCustomCallDescriptor>(opaque, opaque_len);

    // const float *pose = reinterpret_cast<const float *> (buffers[0]);
    // const float *pos = reinterpret_cast<const float *> (buffers[1]);
    // const int *tri = reinterpret_cast<const int *> (buffers[2]);
    // const int *_ranges = reinterpret_cast<const int *> (buffers[3]);
    // const float *projectionMatrix = reinterpret_cast<const float *> (buffers[4]);
    // const int *_resolution = reinterpret_cast<const int *> (buffers[5]);
    
    // float *out = reinterpret_cast<float *> (buffers[6]);
    // float *out_db = reinterpret_cast<float *> (buffers[7]);

    // const float *dy = reinterpret_cast<const float *> (buffers[8]);
    // const float *ddb = reinterpret_cast<const float *> (buffers[9]);

    // float *grad = reinterpret_cast<float *> (buffers[10]);  // output
    // cudaMemset(grad, 0, d.num_images*d.num_vertices*4*sizeof(float));

    // auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);

    // cudaStreamSynchronize(stream);

    // RasterizeGradParams p;
    // bool enable_db = true;

    // // Determine instance mode.
    // p.instance_mode = 1;
    // NVDR_CHECK(p.instance_mode == 1, "Should be in instance mode; check input sizes");

    // // Shape is taken from the rasterizer output tensor.
    // p.depth  = d.num_images;
    // p.height = d.height;
    // p.width  = d.width
    // NVDR_CHECK(p.depth > 0 && p.height > 0 && p.width > 0, "resolution must be [>0, >0, >0]");

    // // Populate parameters.
    // p.numTriangles = d.num_triangles
    // p.numVertices = d.num_vertices;
    // p.pose = pose;
    // p.pos = pos;
    // p.tri = tri;
    // p.out = rast_out;
    // p.dy  = dy;
    // p.ddb = enable_db ? ddb : NULL;

    // // Set up pixel position to clip space x, y transform.
    // p.xs = 2.f / (float)p.width;
    // p.xo = 1.f / (float)p.width - 1.f;
    // p.ys = 2.f / (float)p.height;
    // p.yo = 1.f / (float)p.height - 1.f;

    // // Output tensor for position gradients.
    // p.grad = grad;

    // // Verify that buffers are aligned to allow float2/float4 operations.
    // NVDR_CHECK(!((uintptr_t)p.pos & 15), "pos input tensor not aligned to float4");
    // NVDR_CHECK(!((uintptr_t)p.dy  &  7), "dy input tensor not aligned to float2");
    // NVDR_CHECK(!((uintptr_t)p.ddb & 15), "ddb input tensor not aligned to float4");

    // // Choose launch parameters.
    // dim3 blockSize = getLaunchBlockSize(RAST_GRAD_MAX_KERNEL_BLOCK_WIDTH, RAST_GRAD_MAX_KERNEL_BLOCK_HEIGHT, p.width, p.height);
    // dim3 gridSize  = getLaunchGridSize(blockSize, p.width, p.height, p.depth);

    // // Launch CUDA kernel to populate gradient values.
    // void* args[] = {&p};
    // enable_db = false;
    // void* func = enable_db ? (void*)RasterizeGradKernelDb : (void*)RasterizeGradKernel;
    // NVDR_CHECK_CUDA_ERROR(cudaLaunchKernel(func, gridSize, blockSize, args, 0, stream));

    // cudaStreamSynchronize(stream);
}
