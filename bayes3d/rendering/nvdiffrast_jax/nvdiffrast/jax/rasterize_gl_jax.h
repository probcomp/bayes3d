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
#include "../common/framework.h"
#include "../common/glutil.h"

//------------------------------------------------------------------------
// OpenGL-related persistent state for forward op.

struct RasterizeGLState // Must be initializable by memset to zero.
{
    int                     width;              // Allocated frame buffer width.
    int                     height;             // Allocated frame buffer height.
    int                     depth;              // Allocated frame buffer depth.
    int                     posCount;           // Allocated position buffer in floats.
    int                     triCount;           // Allocated triangle buffer in ints.
    GLContext               glctx;
    GLuint                  glFBO;
    GLuint                  glColorBuffer[2];
    GLuint                  glPrevOutBuffer;
    GLuint                  glDepthStencilBuffer;
    GLuint                  glVAO;
    GLuint                  glTriBuffer;
    GLuint                  glPosBuffer;
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
    int                     enableDB;
    int                     enableZModify;      // Modify depth in shader, workaround for a rasterization issue on A100.
};


class RasterizeGLStateWrapper;

struct SetUpCustomCallDescriptor {
    RasterizeGLStateWrapper* gl_state_wrapper;
    
    int height;
    int width;
    int num_layers;
};

struct LoadVerticesCustomCallDescriptor {
    RasterizeGLStateWrapper* gl_state_wrapper;
    long num_vertices;
    long num_triangles;
};

struct RasterizeCustomCallDescriptor {
    RasterizeGLStateWrapper* gl_state_wrapper;
    float proj[16];
    int num_objects;
    int num_images;
    int on_object;
};

struct DiffRasterizeCustomCallDescriptor {
    RasterizeGLStateWrapper* gl_state_wrapper;
    int num_images;
    int num_vertices;
    int num_triangles;
};

#include <string>

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(
      std::is_trivially_constructible<To>::value,
      "This implementation additionally requires destination type to be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

// Note that bit_cast is only available in recent C++ standards so you might need
// to provide a shim like the one in lib/kernel_helpers.h
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

#include <pybind11/pybind11.h>

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

//------------------------------------------------------------------------
// Shared C++ code prototypes.

void rasterizeInitGLContext(NVDR_CTX_ARGS, RasterizeGLState& s, int cudaDeviceIdx);
void rasterizeResizeBuffers(NVDR_CTX_ARGS, RasterizeGLState& s, bool& changes, int posCount, int triCount, int width, int height, int depth);
void rasterizeRender(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, const float* posPtr, int posCount, int vtxPerInstance, const int32_t* triPtr, int triCount, const int32_t* rangesPtr, int width, int height, int depth, int peeling_idx);
void rasterizeCopyResults(NVDR_CTX_ARGS, RasterizeGLState& s, cudaStream_t stream, float** outputPtr, int width, int height, int depth);
void rasterizeReleaseBuffers(NVDR_CTX_ARGS, RasterizeGLState& s);

//------------------------------------------------------------------------
#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))
