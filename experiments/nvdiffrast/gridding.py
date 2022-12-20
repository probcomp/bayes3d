import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3.viz
from jax3dp3.rendering import render_planes
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    
)
import os
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos, depth_to_coords_in_camera

from jax.scipy.stats.multivariate_normal import logpdf
from jax.scipy.special import logsumexp

from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.viz import save_depth_image
import jax3dp3.transforms_3d as t3d
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import jax3dp3.distributions
from jax3dp3.enumerations import fibonacci_sphere, geodesicHopf_select_axis
from jax3dp3.enumerations import make_translation_grid_enumeration, make_rotation_grid_enumeration
import torch
import matplotlib.pyplot as plt
import cv2
from jax3dp3.enumerations import make_translation_grid_enumeration, make_rotation_grid_enumeration
import jax3dp3.nvdiffrast.common as dr
from jax3dp3.camera import open_gl_projection_matrix
import trimesh

def tensor(a, **kwargs):
    return torch.tensor(np.array(a), device='cuda', **kwargs)

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
vertices = tensor(np.array(mesh.vertices,dtype="f"))
view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)
triangles = tensor(mesh.faces , dtype=torch.int32)
dr.load_vertices(glenv, view_space_vertices_h, triangles, h,w)

obs_image = tensor(np.zeros((h, w, 4),dtype="f"))
dr.load_obs_image(glenv, obs_image)

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.0]))
variance = 0.5
concentration = 0.01
key = jax.random.PRNGKey(30)
sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)
gt_pose = sampler_jit(key, center_of_sampling, variance, concentration)
poses = tensor(np.array([gt_pose for _ in range(1)]))
gt_image = dr.rasterize(glenv, poses, proj_list, h,w, False)[0]
save_depth_image(gt_image[:,:,2].cpu(), "gt_image.png", max=max_depth)

dr.load_obs_image(glenv, gt_image)

image_from_pose = lambda pose: dr.rasterize(glenv, pose, proj_list, h,w, True)
likelihoods_from_pose = lambda pose: dr.rasterize(glenv, pose, proj_list, h,w, True)[:,:,:,-1].sum((1,2))

translation_deltas_1 = tensor(make_translation_grid_enumeration(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 5, 5, 5))
translation_deltas_2 = tensor(make_translation_grid_enumeration(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 5, 5, 5))
translation_deltas_3 = tensor(make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5))
rotation_deltas = tensor(make_rotation_grid_enumeration(20, 20))

initial_pose_estimate = tensor(sampler_jit(jax.random.split(key)[0], center_of_sampling, variance, concentration))
x = initial_pose_estimate

proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_1)
weights_new = likelihoods_from_pose(proposals)
x = proposals[torch.argmax(weights_new)]

proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas)
weights_new = image_from_pose(proposals)[:,:,:,-1].sum((1,2))
x = proposals[torch.argmax(weights_new)]

proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_2)
weights_new = image_from_pose(proposals)[:,:,:,-1].sum((1,2))
x = proposals[torch.argmax(weights_new)]

proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas)
weights_new = image_from_pose(proposals)[:,:,:,-1].sum((1,2))
x = proposals[torch.argmax(weights_new)]

proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_3)
weights_new = image_from_pose(proposals)[:,:,:,-1].sum((1,2))
x = proposals[torch.argmax(weights_new)]

proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas)
weights_new = image_from_pose(proposals)[:,:,:,-1].sum((1,2))
x = proposals[torch.argmax(weights_new)]


gt_img_viz = jax3dp3.viz.get_depth_image(gt_image[:,:,2].cpu(),max=max_depth) 
initial_viz = jax3dp3.viz.get_depth_image(image_from_pose(x[None,...])[0,:,:,2].cpu(),max=max_depth) 
final_viz = jax3dp3.viz.get_depth_image(image_from_pose(x[None,...])[0,:,:,2].cpu(),max=max_depth) 

jax3dp3.viz.multi_panel(
    [gt_img_viz, initial_viz, final_viz],
    ["GT", "Start", "Pred"],
    10,
    100,
    20
).save("all_estimates.png")

from IPython import embed; embed()

