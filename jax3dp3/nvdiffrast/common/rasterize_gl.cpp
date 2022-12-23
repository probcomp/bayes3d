// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "rasterize_gl.h"
#include "glutil.h"
#include "torch_common.inl"
#include "torch_types.h"
#include "common.h"
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <chrono>

#include <vector>
#include <iostream>
#define STRINGIFY_SHADER_SOURCE(x) #x

//------------------------------------------------------------------------
// Helpers.

#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
static int ROUND_UP_BITS(uint32_t x, uint32_t y)
{
    // Round x up so that it has at most y bits of mantissa.
    if (x < (1u << y))
        return x;
    uint32_t m = 0;
    while (x & ~m)
        m = (m << 1) | 1u;
    m >>= y;
    if (!(x & m))
        return x;
    return (x | m) + 1u;
}

//------------------------------------------------------------------------
// Draw command struct used by rasterizer.

struct GLDrawCmd
{
    uint32_t    count;
    uint32_t    instanceCount;
    uint32_t    firstIndex;
    uint32_t    baseVertex;
    uint32_t    baseInstance;
};

//------------------------------------------------------------------------
// GL helpers.

static void compileGLShader(NVDR_CTX_ARGS, const RasterizeGLState& s, GLuint* pShader, GLenum shaderType, const char* src_buf)
{
    std::string src(src_buf);

    // Set preprocessor directives.
    int n = src.find('\n') + 1; // After first line containing #version directive.
    if (s.enableZModify)
        src.insert(n, "#define IF_ZMODIFY(x) x\n");
    else
        src.insert(n, "#define IF_ZMODIFY(x)\n");

    const char *cstr = src.c_str();
    *pShader = 0;
    NVDR_CHECK_GL_ERROR(*pShader = glCreateShader(shaderType));
    NVDR_CHECK_GL_ERROR(glShaderSource(*pShader, 1, &cstr, 0));
    NVDR_CHECK_GL_ERROR(glCompileShader(*pShader));
}

static void constructGLProgram(NVDR_CTX_ARGS, GLuint* pProgram, GLuint glVertexShader, GLuint glFragmentShader)
{
    *pProgram = 0;

    GLuint glProgram = 0;
    NVDR_CHECK_GL_ERROR(glProgram = glCreateProgram());
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, glVertexShader));
    NVDR_CHECK_GL_ERROR(glAttachShader(glProgram, glFragmentShader));
    NVDR_CHECK_GL_ERROR(glLinkProgram(glProgram));

    GLint linkStatus = 0;
    NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_LINK_STATUS, &linkStatus));
    if (!linkStatus)
    {
        GLint infoLen = 0;
        NVDR_CHECK_GL_ERROR(glGetProgramiv(glProgram, GL_INFO_LOG_LENGTH, &infoLen));
        if (infoLen)
        {
            const char* hdr = "glLinkProgram() failed:\n";
            std::vector<char> info(strlen(hdr) + infoLen);
            strcpy(&info[0], hdr);
            NVDR_CHECK_GL_ERROR(glGetProgramInfoLog(glProgram, infoLen, &infoLen, &info[strlen(hdr)]));
            NVDR_CHECK(0, &info[0]);
        }
        NVDR_CHECK(0, "glLinkProgram() failed");
    }

    *pProgram = glProgram;
}

//------------------------------------------------------------------------
// Shared C++ functions.

void rasterizeInitGLContext(NVDR_CTX_ARGS, RasterizeGLState& s, int cudaDeviceIdx)
{
    // Create GL context and set it current.
    s.glctx = createGLContext(cudaDeviceIdx);
    setGLContext(s.glctx);

    // Version check.
    GLint vMajor = 0;
    GLint vMinor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &vMajor);
    glGetIntegerv(GL_MINOR_VERSION, &vMinor);
    glGetError(); // Clear possible GL_INVALID_ENUM error in version query.
    LOG(INFO) << "OpenGL version reported as " << vMajor << "." << vMinor;
    NVDR_CHECK((vMajor == 4 && vMinor >= 4) || vMajor > 4, "OpenGL 4.4 or later is required");

    // Enable depth modification workaround on A100 and later.
    int capMajor = 0;
    NVDR_CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&capMajor, cudaDevAttrComputeCapabilityMajor, cudaDeviceIdx));
    s.enableZModify = (capMajor >= 8);

    // Number of output buffers.
    int num_outputs = s.enableDB ? 2 : 1;

    // Set up vertex shader.
    compileGLShader(NVDR_CTX_PARAMS, s, &s.glVertexShader, GL_VERTEX_SHADER,
        "#version 330\n"
        "#extension GL_ARB_shader_draw_parameters : enable\n"
        "#extension GL_ARB_explicit_uniform_location : enable\n"
        "#extension GL_AMD_vertex_shader_layer : enable\n"
        STRINGIFY_SHADER_SOURCE(
            layout(location = 0) uniform mat4 mvp;
            in vec4 in_vert;
            out vec4 vertex_on_object;
            uniform sampler2D texture;
            void main()
            {
                gl_Layer = gl_DrawIDARB;
                vec4 v1 = texelFetch(texture, ivec2(0, gl_Layer), 0);
                vec4 v2 = texelFetch(texture, ivec2(1, gl_Layer), 0);
                vec4 v3 = texelFetch(texture, ivec2(2, gl_Layer), 0);
                vec4 v4 = texelFetch(texture, ivec2(3, gl_Layer), 0);
                mat4 pose_mat = transpose(mat4(v1,v2,v3,v4));
                vertex_on_object = pose_mat * in_vert;
                gl_Position = mvp * vertex_on_object;
            }
        )
    );

    // Fragment shader without bary differential output.
    compileGLShader(NVDR_CTX_PARAMS, s, &s.glFragmentShader, GL_FRAGMENT_SHADER,
        "#version 430\n"
        STRINGIFY_SHADER_SOURCE(
            in vec4 vertex_on_object;
            out vec4 fragColor;
            void main()
            {
                fragColor = vec4(vertex_on_object);
            }
        )
    );

    // Finalize programs.
    constructGLProgram(NVDR_CTX_PARAMS, &s.glProgram, s.glVertexShader, s.glFragmentShader);
    constructGLProgram(NVDR_CTX_PARAMS, &s.glProgramDP, s.glVertexShader, s.glFragmentShader);

    // Construct main fbo and bind permanently.
    NVDR_CHECK_GL_ERROR(glGenFramebuffers(1, &s.glFBO));
    NVDR_CHECK_GL_ERROR(glBindFramebuffer(GL_FRAMEBUFFER, s.glFBO));

    // Enable two color attachments.
    GLenum draw_buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    NVDR_CHECK_GL_ERROR(glDrawBuffers(num_outputs, draw_buffers));

    // Construct vertex array object.
    NVDR_CHECK_GL_ERROR(glGenVertexArrays(1, &s.glVAO));
    NVDR_CHECK_GL_ERROR(glBindVertexArray(s.glVAO));

    // Construct position buffer, bind permanently, enable, set ptr.
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glPosBuffer));
    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_ARRAY_BUFFER, s.glPosBuffer));
    NVDR_CHECK_GL_ERROR(glEnableVertexAttribArray(0));
    NVDR_CHECK_GL_ERROR(glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0));

    // Construct index buffer and bind permanently.
    NVDR_CHECK_GL_ERROR(glGenBuffers(1, &s.glTriBuffer));
    NVDR_CHECK_GL_ERROR(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s.glTriBuffer));

    // Set up depth test.
    NVDR_CHECK_GL_ERROR(glEnable(GL_DEPTH_TEST));
    NVDR_CHECK_GL_ERROR(glDepthFunc(GL_LESS));
    NVDR_CHECK_GL_ERROR(glClearDepth(1.0));

    // Create and bind output buffers. Storage is allocated later.
    NVDR_CHECK_GL_ERROR(glGenTextures(num_outputs, s.glColorBuffer));
    for (int i=0; i < num_outputs; i++)
    {
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer[i]));
        NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, s.glColorBuffer[i], 0));
    }

    // Create and bind depth/stencil buffer. Storage is allocated later.
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glDepthStencilBuffer));
    NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
    NVDR_CHECK_GL_ERROR(glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, s.glDepthStencilBuffer, 0));

    // Create texture name for previous output buffer (depth peeling).
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glPrevOutBuffer));
}
void rasterizeReleaseBuffers(NVDR_CTX_ARGS, RasterizeGLState& s)
{
    int num_outputs = s.enableDB ? 2 : 1;

    if (s.cudaPosBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPosBuffer));
        s.cudaPosBuffer = 0;
    }

    if (s.cudaTriBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriBuffer));
        s.cudaTriBuffer = 0;
    }

    for (int i=0; i < num_outputs; i++)
    {
        if (s.cudaColorBuffer[i])
        {
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer[i]));
            s.cudaColorBuffer[i] = 0;
        }
    }

    if (s.cudaPrevOutBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPrevOutBuffer));
        s.cudaPrevOutBuffer = 0;
    }
}



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

void threedp3_likelihood(float *pos, float *latent_points, float *likelihoods, float *obs_image, float r, int width, int height, int depth);

void load_vertices_fwd(RasterizeGLStateWrapper& stateWrapper, torch::Tensor pos, torch::Tensor tri, int height, int width)
{
    int depth = 1024;
    const at::cuda::OptionalCUDAGuard device_guard(device_of(pos));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RasterizeGLState& s = *stateWrapper.pState;

    // Check inputs.
    NVDR_CHECK_DEVICE(pos, tri);
    NVDR_CHECK_CONTIGUOUS(pos, tri);
    NVDR_CHECK_F32(pos);
    NVDR_CHECK_I32(tri);

    // Check that GL context was created for the correct GPU.
    NVDR_CHECK(pos.get_device() == stateWrapper.cudaDeviceIdx, "GL context must must reside on the same device as input tensors");

    // Determine number of outputs

    // Determine instance mode and check input dimensions.
    NVDR_CHECK(pos.sizes().size() == 2 && pos.size(0) > 0 && pos.size(1) == 4, "range mode - pos must have shape [>0, 4]");
    NVDR_CHECK(tri.sizes().size() == 2 && tri.size(0) > 0 && tri.size(1) == 3, "tri must have shape [>0, 3]");

    // Get output shape.
    NVDR_CHECK(height > 0 && width > 0, "resolution must be [>0, >0]");

    // Get position and triangle buffer sizes in int32/float32.
    int posCount = 4 * pos.size(0);
    int triCount = 3 * tri.size(0);

    // Set the GL context unless manual context.
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    // Resize all buffers.

    // Resize vertex buffer?
    if (s.cudaPosBuffer)
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPosBuffer));
    s.posCount = (posCount > 64) ? ROUND_UP_BITS(posCount, 2) : 64;
    LOG(INFO) << "Increasing position buffer size to " << s.posCount << " float32";
    NVDR_CHECK_GL_ERROR(glBufferData(GL_ARRAY_BUFFER, s.posCount * sizeof(float), NULL, GL_DYNAMIC_DRAW));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaPosBuffer, s.glPosBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

    // Resize triangle buffer?
    if (s.cudaTriBuffer)
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaTriBuffer));
    s.triCount = (triCount > 64) ? ROUND_UP_BITS(triCount, 2) : 64;
    LOG(INFO) << "Increasing triangle buffer size to " << s.triCount << " int32";
    NVDR_CHECK_GL_ERROR(glBufferData(GL_ELEMENT_ARRAY_BUFFER, s.triCount * sizeof(int32_t), NULL, GL_DYNAMIC_DRAW));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(&s.cudaTriBuffer, s.glTriBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

    int num_outputs = 1;
    if (s.cudaColorBuffer[0])
        for (int i=0; i < num_outputs; i++)
            NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaColorBuffer[i]));

    if (s.cudaPrevOutBuffer)
    {
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPrevOutBuffer));
        s.cudaPrevOutBuffer = 0;
    }

    // New framebuffer size.
    s.width  = (width > s.width) ? width : s.width;
    s.height = (height > s.height) ? height : s.height;
    s.depth  = (depth > s.depth) ? depth : s.depth;
    s.width  = ROUND_UP(s.width, 32);
    s.height = ROUND_UP(s.height, 32);
    std::cout << "Increasing frame buffer size to (width, height, depth) = (" << s.width << ", " << s.height << ", " << s.depth << ")" << std::endl;

    // Allocate color buffers.
    for (int i=0; i < num_outputs; i++)
    {
        NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glColorBuffer[i]));
        NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, s.width, s.height, s.depth, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    }

    // Allocate depth/stencil buffer.
    NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D_ARRAY, s.glDepthStencilBuffer));
    NVDR_CHECK_GL_ERROR(glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8, s.width, s.height, s.depth, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0));

    // (Re-)register all GL buffers into Cuda.
    for (int i=0; i < num_outputs; i++)
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaColorBuffer[i], s.glColorBuffer[i], GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));


    const float* posPtr = pos.data_ptr<float>();
    const int32_t* triPtr = tri.data_ptr<int32_t>();
    int vtxPerInstance = pos.size(1);

    // Copy both position and triangle buffers.
    void* glPosPtr = NULL;
    void* glTriPtr = NULL;
    size_t posBytes = 0;
    size_t triBytes = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(2, &s.cudaPosBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glPosPtr, &posBytes, s.cudaPosBuffer));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&glTriPtr, &triBytes, s.cudaTriBuffer));
    NVDR_CHECK(posBytes >= posCount * sizeof(float), "mapped GL position buffer size mismatch");
    NVDR_CHECK(triBytes >= triCount * sizeof(int32_t), "mapped GL triangle buffer size mismatch");
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glPosPtr, posPtr, posCount * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    NVDR_CHECK_CUDA_ERROR(cudaMemcpyAsync(glTriPtr, triPtr, triCount * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(2, &s.cudaPosBuffer, stream));

}

void load_obs_image(RasterizeGLStateWrapper& stateWrapper,  torch::Tensor obs_image){
    NVDR_CHECK_DEVICE(obs_image);
    NVDR_CHECK_CONTIGUOUS(obs_image);
    NVDR_CHECK_F32(obs_image);
    RasterizeGLState& s = *stateWrapper.pState;
    unsigned int bytes = obs_image.size(0)*obs_image.size(1)*4*sizeof(float);
    cudaMalloc((void**)&s.obs_image, bytes);
    cudaMemcpy(s.obs_image, obs_image.data_ptr<float>(),  bytes, cudaMemcpyDeviceToDevice);
}

torch::Tensor rasterize_fwd_gl(RasterizeGLStateWrapper& stateWrapper,  torch::Tensor pose, const std::vector<float>& proj, uint height, uint width, float r)
{
    NVDR_CHECK_DEVICE(pose);
    NVDR_CHECK_CONTIGUOUS(pose);
    NVDR_CHECK_F32(pose);

    auto start = std::chrono::high_resolution_clock::now();

    // const at::cuda::OptionalCUDAGuard device_guard(device_of(pos));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RasterizeGLState& s = *stateWrapper.pState;

    // Set the GL context unless manual context.
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    uint num_total_poses = pose.size(0);
    uint sub_total_poses = 1024;

    uint pose_bytes = sub_total_poses*16*sizeof(float);
    const float* posePtr = pose.data_ptr<float>();


    unsigned long int image_bytes= num_total_poses*height*width*4*sizeof(float);




    // Set the GL context unless manual context.
    if (stateWrapper.automatic)
        setGLContext(s.glctx);

    NVDR_CHECK_GL_ERROR(glUseProgram(s.glProgram));
    glUniformMatrix4fv(0, 1, GL_TRUE, &proj[0]);

    if (s.cudaPoseTexture)
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(s.cudaPoseTexture));
    NVDR_CHECK_GL_ERROR(glGenTextures(1, &s.glPoseTexture));
    NVDR_CHECK_GL_ERROR(glBindTexture(GL_TEXTURE_2D, s.glPoseTexture));
    NVDR_CHECK_GL_ERROR(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, sub_total_poses, 0, GL_RGBA, GL_FLOAT, 0));
    NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    NVDR_CHECK_GL_ERROR(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));


    std::vector<GLDrawCmd> drawCmdBuffer(sub_total_poses);
    for (int i=0; i < sub_total_poses; i++)
    {
        GLDrawCmd& cmd = drawCmdBuffer[i];
        cmd.firstIndex    = 0;
        cmd.count         = s.triCount;
        cmd.baseVertex    = 0;
        cmd.baseInstance  = 0;
        cmd.instanceCount = 1;
    }



    // Copy color buffers to output tensors.
    cudaArray_t array = 0;
    cudaChannelFormatDesc arrayDesc = {};   // For error checking.
    cudaExtent arrayExt = {};               // For error checking.
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, s.cudaColorBuffer, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer[0], 0, 0));
    NVDR_CHECK_CUDA_ERROR(cudaArrayGetInfo(&arrayDesc, &arrayExt, NULL, array));
    NVDR_CHECK(arrayDesc.f == cudaChannelFormatKindFloat, "CUDA mapped array data kind mismatch!!");
    NVDR_CHECK(arrayDesc.x == 32 && arrayDesc.y == 32 && arrayDesc.z == 32 && arrayDesc.w == 32, "CUDA mapped array data width mismatch");
    // NVDR_CHECK(arrayExt.width >= width && arrayExt.height >= height && arrayExt.depth >= num_total_poses, "CUDA mapped array extent mismatch!!");
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, s.cudaColorBuffer, stream));

    NVDR_CHECK_CUDA_ERROR(cudaGraphicsGLRegisterImage(&s.cudaPoseTexture, s.glPoseTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    cudaArray_t pose_array = 0;
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaPoseTexture, stream));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&pose_array, s.cudaPoseTexture, 0, 0));
    NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaPoseTexture, stream));

    // std::cout << height << " " << width << " " << num_images << " after !!" << std::endl;
    torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output_torch_tensor = torch::empty({num_total_poses,height,width,4}, opts);

    for(int start_pose_idx=0; start_pose_idx < num_total_poses; start_pose_idx+=sub_total_poses)
    {
        auto poses_on_this_iter = std::min(num_total_poses-start_pose_idx, sub_total_poses);
        // Set viewport, clear color buffer(s) and depth/stencil buffer.
        NVDR_CHECK_GL_ERROR(glViewport(0, 0, width, height));
        NVDR_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

        NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &s.cudaPoseTexture, stream));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&pose_array, s.cudaPoseTexture, 0, 0));
        NVDR_CHECK_CUDA_ERROR(cudaMemcpyToArrayAsync(pose_array, 0, 0, posePtr + start_pose_idx*16, pose_bytes, cudaMemcpyDeviceToDevice));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &s.cudaPoseTexture, stream));

        // Draw!
        NVDR_CHECK_GL_ERROR(glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, &drawCmdBuffer[0], poses_on_this_iter, sizeof(GLDrawCmd)));

        NVDR_CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, s.cudaColorBuffer, stream));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&array, s.cudaColorBuffer[0], 0, 0));
        cudaMemcpy3DParms p = {0};
        p.srcArray = array;
        p.dstPtr.ptr = output_torch_tensor.data_ptr<float>() + start_pose_idx*height*width*4;
        p.dstPtr.pitch = width * 4 * sizeof(float);
        p.dstPtr.xsize = width;
        p.dstPtr.ysize = height;
        p.extent.width = width;
        p.extent.height = height;
        p.extent.depth = poses_on_this_iter;
        p.kind = cudaMemcpyDeviceToDevice;
        NVDR_CHECK_CUDA_ERROR(cudaMemcpy3D(&p));
        NVDR_CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, s.cudaColorBuffer, stream));
    }
    // torch::Tensor torch_out =  torch::empty({num_total_poses,height,width,4}, opts);
    // cudaMemcpy(torch_out.data_ptr<float>(), output_torch_tensor.data_ptr<float>(), num_total_poses*height*width*4*4, cudaMemcpyDeviceToDevice);

    // torch::Tensor prev_tensor = torch::empty({num_total_poses,height,width,4}, opts);
    // cudaMemcpy(prev_tensor.data_ptr<float>(), output_image_cuda, image_bytes, cudaMemcpyDeviceToDevice);
    // output_torch_tensor = torch::sum(prev_tensor, std::vector<int64_t>({1,2,3}));
    // }
    // else{
    //     output_torch_tensor = torch::empty({num_total_poses,height,width,4}, opts);
    //     cudaMemcpy(output_torch_tensor.data_ptr<float>(), output_image_cuda, image_bytes, cudaMemcpyDeviceToDevice);


    // }
    // float* output_ptr = output_torch_tensor.data_ptr<float>();
    // float temp = 1.0;
    // cudaMemcpy(&temp, output_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << temp << std::endl;
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start) * 1e-6;
    std::cout << "duration : " << duration.count() << std::endl;


    // Done. Release GL context and return.
    if (stateWrapper.automatic)
        releaseGLContext();

    return output_torch_tensor;
}

std::vector<float> rasterize_get_best_pose_fwd(RasterizeGLStateWrapper& stateWrapper,  torch::Tensor pose, const std::vector<float>& proj, uint height, uint width, float r)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RasterizeGLState& s = *stateWrapper.pState;
    auto start2 = std::chrono::high_resolution_clock::now();

    torch::Tensor output_torch_tensor = rasterize_fwd_gl(stateWrapper, pose, proj, height, width, r);

    uint num_total_poses = pose.size(0);
    float* likelihoods;
    cudaMalloc((void**)&likelihoods, num_total_poses*sizeof(float));
    std::vector<float> likelihoods_cpu(num_total_poses);


    dim3 blockSize(1024, 1, 1);
    dim3 gridSize(height,width, 1);
    torch::Tensor num_latent_points = output_torch_tensor.index({"...", 3}).sum({1,2});
    std::cout << num_latent_points.sizes() << std::endl;
    float *output_ptr = output_torch_tensor.data_ptr<float>();
    float *num_latent_points_ptr = num_latent_points.data_ptr<float>();
    void* args[] = {&output_ptr, &num_latent_points_ptr, &likelihoods, &s.obs_image, &r, &width, &height, &num_total_poses};
    // std::cout << height << " " << width << " " << num_images << " after !!" << std::endl;

    cudaLaunchKernel((void*)threedp3_likelihood, gridSize, blockSize, args, 0, stream);

    float temp = 1.0;
    cudaMemcpy(&temp, output_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << temp << std::endl;
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);    
    std::cout << "duration 2: " << duration2.count() << std::endl;
    


    auto start3 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&likelihoods_cpu[0], likelihoods, num_total_poses*sizeof(float), cudaMemcpyDeviceToHost);
    auto stop3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(stop3 - start3);    
    std::cout << "duration 3: " << duration3.count() << std::endl;

    return likelihoods_cpu;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeGLStateWrapper>(m, "RasterizeGLStateWrapper").def(pybind11::init<bool, bool, int>())
        .def("set_context",     &RasterizeGLStateWrapper::setContext)
        .def("release_context", &RasterizeGLStateWrapper::releaseContext);

    // Ops.
    m.def("load_vertices_fwd", &load_vertices_fwd, "rasterize forward op (opengl)");
    m.def("load_obs_image", &load_obs_image, "rasterize forward op (opengl)");
    m.def("rasterize_fwd_gl", &rasterize_fwd_gl, "rasterize forward op (opengl)");
    m.def("rasterize_get_best_pose_fwd", &rasterize_get_best_pose_fwd, "rasterize forward op (opengl)");
}

//------------------------------------------------------------------------


