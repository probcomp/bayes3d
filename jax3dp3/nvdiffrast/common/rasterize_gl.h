// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

//------------------------------------------------------------------------
// Do not try to include OpenGL stuff when compiling CUDA kernels for torch.

#if !(defined(NVDR_TORCH) && defined(__CUDACC__))
#include "framework.h"
#include "glutil.h"

//------------------------------------------------------------------------
// OpenGL-related persistent state for forward op.

struct RasterizeGLState // Must be initializable by memset to zero.
{
    int                     width;              // Allocated frame buffer width.
    int                     height;             // Allocated frame buffer height.
    int                     depth;              // Allocated frame buffer depth.
    int                     img_width;              // Allocated frame buffer depth.
    int                     img_height;              // Allocated frame buffer depth.
    uint                     num_layers;              // Allocated frame buffer depth.
    std::vector<float>      proj;
    int                     posCount;           // Allocated position buffer in floats.
    int                     triCounts[1000];           // Allocated triangle buffer in ints.
    int                     model_counter;           // Allocated triangle buffer in ints.
    GLContext               glctx;
    GLuint                  glFBO;
    GLuint                  glColorBuffer[2];
    GLuint                  glPrevOutBuffer;
    GLuint                  glDepthStencilBuffer;
    GLuint                  glVAOs[100];
    GLuint                  glTriBuffer;
    GLuint                  glPosBuffer;
    GLuint                  glPoseTexture;
    GLuint                  glProgram;
    GLuint                  glProgramDP;
    GLuint                  glVertexShader;
    GLuint                  glGeometryShader;
    GLuint                  glFragmentShader;
    GLuint                  glFragmentShaderDP;
    cudaGraphicsResource_t  cudaColorBuffer[2];
    cudaGraphicsResource_t  cudaPrevOutBuffer;
    cudaGraphicsResource_t  cudaPosBuffer;
    cudaGraphicsResource_t  cudaTriBuffer;
    cudaGraphicsResource_t  cudaPoseTexture;
    cudaArray_t             cuda_color_buffer;
    cudaArray_t             cuda_pose_buffer;
    float*                  obs_image;
    int                     enableDB;
    int                     enableZModify;      // Modify depth in shader, workaround for a rasterization issue on A100.
};

//------------------------------------------------------------------------
// Shared C++ code prototypes.

//------------------------------------------------------------------------
#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
