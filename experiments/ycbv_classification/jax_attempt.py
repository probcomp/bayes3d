import trimesh
import os
import numpy as np
import jax3dp3
import jax3dp3.utils
import jax3dp3.camera
import jax3dp3.nvdiffrast.common as dr
import torch
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax
import jax3dp3.distributions
import jax3dp3.enumerations 

h, w = 200,200
fx,fy = 200, 200
cx,cy = 100, 100
near=0.01
far=100.0
max_depth=5.0
proj_list = list(jax3dp3.camera.open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(-1))

glenv = dr.RasterizeGLContext(output_db=False)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"models/011_banana/textured_simple.obj"))
vertices = np.array(mesh.vertices)
vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
triangles = np.array(mesh.faces)

dr.load_vertices(glenv, torch.tensor(vertices.astype("f"), device='cuda'), torch.tensor(triangles.astype(np.int32), device='cuda'), h,w)

obs_image = jnp.zeros((h,w,4))
dr.load_obs_image(glenv, torch.tensor(np.array(obs_image).astype("f"), device='cuda'))

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.5]))
variance = 0.0000001
concentration = 0.01
key = jax.random.PRNGKey(30)
sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)
gt_pose = sampler_jit(key, center_of_sampling, variance, concentration)
gt_pose_torch = torch.tensor(np.array(gt_pose), device='cuda')
gt_image = dr.rasterize(glenv, gt_pose_torch[None, ...], proj_list, h,w, False)[0]
jax3dp3.viz.save_depth_image(gt_image[:,:,2].cpu(), "gt_image.png", max=max_depth)
dr.load_obs_image(glenv, gt_image)

rotation_deltas = jax3dp3.enumerations.make_rotation_grid_enumeration(50, 20)
poses_to_score = jnp.einsum("ij,ajk->aik", gt_pose, rotation_deltas)

images = dr.rasterize(glenv, torch.tensor(np.array(poses_to_score), device='cuda'), proj_list, h,w, False)
jax3dp3.viz.save_depth_image(images[10,:,:,2].cpu(), "1.png", max=max_depth)

scores = dr.rasterize_get_best_pose(glenv, torch.tensor(np.array(poses_to_score), device='cuda'), proj_list, h,w, 0.5)


from IPython import embed; embed()
