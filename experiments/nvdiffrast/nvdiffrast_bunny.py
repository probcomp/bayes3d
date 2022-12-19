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
import jax3dp3.nvdiffrast.common as dr
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
from jax3dp3.camera import open_gl_projection_matrix
from jax3dp3.likelihood import threedp3_likelihood

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')


h, w = 200,200
cx = (w-1)/2
cy = (h-1)/2
fx = 200.0
fy = 200.0
near=0.05
far=30.0

max_depth = 30.0
glenv = dr.RasterizeGLContext(output_db=False)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 3.2]))
view_space_vertices = t3d.apply_transform(vertices, pose)
vertices = tensor(np.array(view_space_vertices))
num_images = 1024
# vertices = vertices.tile((num_images,1,1))
triangles = tensor(mesh.faces , dtype=torch.int32)


start = time.time()
proj = open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far)
proj_list = list(proj.reshape(-1))
print(proj_list)
view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)
# clip_space_vertices = torch.einsum("ij,abj->abi", proj, view_space_vertices_h).contiguous()

pose = np.array([np.eye(4) for _ in range(1024)])
pose[:,:3,3] = np.array([-1.0, -1.0, 3.0])
pose[:,2,3] = np.linspace(1.0, 10.0, 1024)
pose_list = list(pose.reshape(-1))
# pose = [0.0 for _ in range(16)]
dr.load_vertices(glenv, proj_list, view_space_vertices_h, triangles, num_images,resolution=[h,w], grad_db=False)
dr.rasterize(glenv, pose_list, proj_list, view_space_vertices_h, triangles, num_images,resolution=[h,w], grad_db=False)
start = time.time()
rast, _ = dr.rasterize(glenv, pose_list, proj_list, view_space_vertices_h, triangles, num_images,resolution=[h,w], grad_db=False)
end = time.time()
print ("Time elapsed:", end - start)

rast_reshaped = rast.reshape(num_images, h, w, 4)
jax3dp3.viz.save_depth_image(rast_reshaped[0,:,:,2].cpu().numpy(), "bunny.png",max=10.0)
jax3dp3.viz.save_depth_image(rast_reshaped[-1,:,:,2].cpu().numpy(), "bunny2.png",max=20.0)


from IPython import embed; embed()

# rast_reshaped = rast.reshape(num_images, h, w, 4)

# a,b,c = 9,111,80
# def get_idx(a,b,c):
#     return a * (h*w*4) + b * (w*4) + c*4

# print(rast_reshaped[a,b,c,:])
# idx = get_idx(a,b,c)
# print(rast[idx:idx+5])
# jax3dp3.viz.save_depth_image(rast_reshaped[2,:,:,0].cpu().numpy(), "bunny.png",max=10.0)
# jax3dp3.viz.save_depth_image((rast_reshaped[2,:,:,2] > 0).cpu().numpy(), "bunny2.png",max=50.0)


# print(rast_reshaped.shape)

# from IPython import embed; embed()

# depth = rast_reshaped[:,:,:,2]
# neg_mask = depth == 0
# depth = 2 * near * far / (far + near - depth * (far - near))
# depth[neg_mask] = 0


# jax3dp3.viz.save_depth_image(depth[0,:,:,].cpu().numpy(), "bunny.png",max=10.0)

# from IPython import embed; embed()
