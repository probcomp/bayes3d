// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#if !(defined(NVDR_TORCH) && defined(__CUDACC__))
#include "../common/framework.h"
#include "../common/glutil.h"

struct DiffInterpolateCustomCallDescriptor {
    int num_images;   // attr[0]
    int num_vertices;  // attr[1]
    int num_attributes;  // attr[2]
    int rast_height;  // rast[1]
    int rast_width;  // rast[2]
    int rast_depth;  // rast[0]
    int num_triangles;  // tri[0]
    int num_diff_attributes;  // diff_attr
};

#endif // !(defined(NVDR_TORCH) && defined(__CUDACC__))