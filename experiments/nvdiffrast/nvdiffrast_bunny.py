# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import sys
import trimesh
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
import time
import os
from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
import jax
import time
import trimesh
from jax3dp3.nv_rendering import render_depth, projection_matrix
from jax3dp3.likelihood import threedp3_likelihood

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')


h, w = 120,160
cx = (w-1)/2
cy = (h-1)/2
fx = 200.0
fy = 200.0
near=0.05
far=10.0

max_depth = 10.0
glenv = dr.RasterizeGLContext(output_db=False)


mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 3.2]))
view_space_vertices = t3d.apply_transform(vertices, pose)
vertices = tensor(np.array([view_space_vertices]))
num_images = 1
vertices = vertices.tile((num_images,1,1))
triangles = tensor(mesh.faces , dtype=torch.int32)

# for i in range(num_images):
#     vertices[i,:,0] -= i*0.0001
point_cloud = render_depth(glenv, vertices, triangles, h,w,fx,fy,cx,cy, near, far)

start = time.time()
rast = render_depth(glenv, vertices, triangles, h,w,fx,fy,cx,cy, near, far)
end = time.time()
print ("Time elapsed:", end - start)
print(rast[0][0][0])

depth = rast[:,:,:,2]
neg_mask = depth == 0
depth = 2 * near * far / (far + near - depth * (far - near))
depth[neg_mask] = 0


jax3dp3.viz.save_depth_image(depth[0,:,:,].cpu().numpy(), "bunny.png",max=10.0)

from IPython import embed; embed()
