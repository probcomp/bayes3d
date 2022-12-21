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
import time
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
    return torch.tensor(a, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')


h, w = 120,160
fx,fy = 200.0, 200.0
cx,cy = 80.0, 60.0
near=0.01
far=100.0
proj_list = list(open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(-1))

r = 0.05
outlier_prob = 0.1

max_depth = 20.0

glenv = dr.RasterizeGLContext(output_db=False)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"035_power_drill/textured_simple.obj"))
vertices = tensor(np.array(mesh.vertices,dtype="f"))
view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)
triangles = tensor(np.array(mesh.faces) , dtype=torch.int32)
dr.load_vertices(glenv, view_space_vertices_h, triangles, h,w)

obs_image = tensor(np.zeros((h, w, 4),dtype="f"))
dr.load_obs_image(glenv, obs_image)

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.0]))
variance = 0.5
concentration = 0.01
key = jax.random.PRNGKey(30)
sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)
gt_pose = tensor(np.array(sampler_jit(key, center_of_sampling, variance, concentration)))
gt_image = dr.rasterize(glenv, gt_pose[None, ...], proj_list, h,w, False)[0]
save_depth_image(gt_image[:,:,2].cpu(), "gt_image.png", max=max_depth)

dr.load_obs_image(glenv, gt_image)

image_from_pose = lambda pose: dr.rasterize(glenv, pose, proj_list, h,w, r)
likelihoods_from_pose = lambda pose: dr.rasterize(glenv, pose, proj_list, h,w, r)[:,:,:,-1].sum((1,2))

translation_deltas_1 = tensor(np.array(make_translation_grid_enumeration(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 9,9,9)))
translation_deltas_2 = tensor(np.array(make_translation_grid_enumeration(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 9,9,9)))
translation_deltas_3 = tensor(np.array(make_translation_grid_enumeration(-0.1, -0.1, -0.1, 0.1, 0.2, 0.1, 9,9,9)))
rotation_deltas = tensor(np.array(make_rotation_grid_enumeration(50, 20)))


initial_pose_estimate = tensor(np.array(sampler_jit(jax.random.split(key)[0], center_of_sampling, variance, concentration)))
x = initial_pose_estimate;

start = time.time()
proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_3).contiguous()
scores = likelihoods_from_pose(proposals)
x = proposals[torch.argmax(scores)]
print(x)
end = time.time()
print ("Time elapsed:", end - start)




start = time.time()
proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_3).contiguous()
imgs = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)
best_idx = np.argmax(imgs)
x = proposals[best_idx]
print(x)
end = time.time()
print ("Time elapsed:", end - start)

from IPython import embed; embed()



rendered_img = image_from_pose(x[None,:,:])
likelihoods_from_pose(x[None,:,:])

threedp3_likelihood(jnp.array(gt_image.cpu().numpy()), jnp.array(rendered_img[0].cpu().numpy()), r, 0.01)







proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas).contiguous()
x = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)
proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_3).contiguous()
x = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)
proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas).contiguous()
x = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)
proposals = torch.einsum("ij,ajk->aik", x, translation_deltas_3).contiguous()
x = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)
proposals = torch.einsum("ij,ajk->aik", x, rotation_deltas).contiguous()
x = dr.rasterize_get_best_pose(glenv, proposals, proj_list, h,w, r)

end = time.time()
print ("Time elapsed:", end - start)

gt_img_viz = jax3dp3.viz.get_depth_image(gt_image[:,:,2].cpu(),max=max_depth) 
initial_viz = jax3dp3.viz.get_depth_image(image_from_pose(initial_pose_estimate[None,...])[0,:,:,2].cpu(),max=max_depth)
img = image_from_pose(x[None,...])[0,:,:,2].cpu()
final_viz = jax3dp3.viz.get_depth_image(img ,max=max_depth) 

final_viz.save("test.png")
jax3dp3.viz.multi_panel(
    [gt_img_viz, initial_viz, final_viz],
    ["GT", "Start", "Pred"],
    10,
    100,
    20
).save("all_estimates.png")

from IPython import embed; embed()

