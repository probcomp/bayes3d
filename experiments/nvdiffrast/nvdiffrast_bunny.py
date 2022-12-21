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
import jax3dp3.distributions

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')


h, w = 200,200
fx,fy = 200, 200
cx,cy = 100, 100
near=0.01
far=100.0
proj_list = list(open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(-1))

r = 0.1
outlier_prob = 0.01

max_depth = 20.0


glenv = dr.RasterizeGLContext(output_db=False)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
vertices = tensor(np.array(vertices,dtype="f"))
view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)

triangles = tensor(mesh.faces , dtype=torch.int32)
proj_list = list(open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(-1))
print(proj_list)

obs_image = tensor(np.zeros((h, w, 4),dtype="f"))
dr.load_obs_image(glenv, obs_image)

num_images = 1024*1

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 8.0]))
variance = 0.5
concentration = 0.01
key = jax.random.PRNGKey(5)
sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)
gt_pose = np.array(sampler_jit(key, center_of_sampling, variance, concentration))
gt_pose = np.array(gt_pose)

translation_deltas_1 = np.array(make_translation_grid_enumeration(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 10, 10, 10))
translation_deltas_1[:,:3,3] += np.array([0.0, 0.0, 10.0]).reshape(1,3)
pose = translation_deltas_1
pose[:,:3,:3] = gt_pose[:3,:3][None,:,:]
# pose = np.array([np.eye(4) for _ in range(num_images)])
# pose[:,:3,3] = np.array([0.0, 0.0, 3.0])
# pose[:,2,3] = np.linspace(3.0, 40.0, num_images)
pose_list = tensor(pose.astype("f"))
# pose = [0.0 for _ in range(16)]

dr.load_vertices(glenv, view_space_vertices_h, triangles, h,w)
start = time.time()
rast = dr.rasterize(glenv, pose_list, proj_list, h,w, False)
log_scores = rast[:,:,:,-1].sum((1,2))
i = log_scores.argmax()
end = time.time()
print ("Time elapsed:", end - start)

jax3dp3.viz.save_depth_image(rast[0,:,:,2].cpu().numpy(), "bunny.png",max=20.0)
jax3dp3.viz.save_depth_image(rast[50,:,:,2].cpu().numpy(), "bunny2.png",max=20.0)



dr.load_obs_image(glenv, rast[50,:,:,:])
rast = dr.rasterize(glenv, pose_list, proj_list, h,w, True)
log_scores = rast[:,:,:,-1].sum((1,2))
print(log_scores.argmax())

num_images = 2*1
pose = np.array([np.eye(4) for _ in range(num_images)])
pose[:,:3,3] = np.array([0.0, 0.0, 3.0])
pose[:,2,3] = np.linspace(3.0, 40.0, num_images)
pose_list = tensor(pose.astype("f"))
rast = dr.rasterize(glenv, pose_list, proj_list, h,w, False)


from IPython import embed; embed()



from IPython import embed; embed()

jax3dp3.viz.save_depth_image(rast[1,:,:,-1].cpu().numpy(), "bunny_count.png",max=121.0)

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
