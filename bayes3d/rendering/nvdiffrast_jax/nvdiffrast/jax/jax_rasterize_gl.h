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
#include "../common/torch_types.h"
#include "../common/torch_common.inl"
#include "../common/common.h"
#include "../common/rasterize.h"
#include "../common/rasterize_gl.h"
#include "bindings.h"

struct DiffRasterizeCustomCallDescriptor {
    RasterizeGLStateWrapper* gl_state_wrapper;
    int num_images;
    int num_vertices;
    int num_triangles;
};
struct DiffRasterizeBwdCustomCallDescriptor {
    int num_images;    // pos[0]
    int num_vertices;  //  pos[1]
    int num_triangles;  // tri[0]
    int rast_height;  // rast[1]
    int rast_width;  // rast[2]
    int rast_depth;  // rast[0]
};

//------------------------------------------------------------------------
// Shared C++ code prototypes.

//------------------------------------------------------------------------
// Op prototypes.

void jax_rasterize_fwd_gl(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_rasterize_bwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_interpolate_fwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_interpolate_bwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

#endif
