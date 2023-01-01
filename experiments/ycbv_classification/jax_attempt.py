import trimesh
import time
import os
import numpy as np
import jax.numpy as jnp
import jax
import torch

import jax3dp3
import jax3dp3.utils
import jax3dp3.camera
import jax3dp3.nvdiffrast.common as dr
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
from jax3dp3.parallel_gl_renderer import setup, load_model, render
import jax3dp3.distributions
import jax3dp3.enumerations
import jax.dlpack
import jax3dp3.likelihood
import jax3dp3.parallel_gl_renderer as pgl
from jax3dp3.single_object_recognition import classify

h, w = 120, 160
fx,fy = 200.0, 200.0
cx,cy = 80.0, 60.0
near=0.01
far=50.0
max_depth=2.0

setup(h, w, fx, fy, cx, cy, near, far)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
for model in model_names:
    model_path = os.path.join(jax3dp3.utils.get_assets_dir(),"models/{}/textured_simple.obj".format(model))
    load_model(model_path, h, w)

gt_model_idx = 19
gt_mesh_name = model_names[gt_model_idx]

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.5]))
variance = 0.0000001
concentration = 0.01
key = jax.random.PRNGKey(10)
sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)
gt_pose = sampler_jit(key, center_of_sampling, variance, concentration)
# gt_pose_torch = torch.tensor(np.array(gt_pose), device='cuda')
gt_image = render(jnp.stack([gt_pose, gt_pose]), h,w,gt_model_idx)[0]
jax3dp3.viz.save_depth_image(gt_image[:,:,2], "gt_image.png", max=max_depth)


r = 0.05
outlier_prob = 0.1
def scorer(rendered_image, gt):
    weight = jax3dp3.likelihood.threedp3_likelihood(gt, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes=(0, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

import jax3dp3.bbox
non_zero_points = gt_image[gt_image[:,:,2]>0,:3]
_, centroid_pose = jax3dp3.bbox.axis_aligned_bounding_box(non_zero_points)
rotation_deltas = jax3dp3.enumerations.make_rotation_grid_enumeration(50, 20)
poses_to_score = jnp.einsum("ij,ajk->aik", centroid_pose, rotation_deltas)

object_indices = list(range(len(model_names)))

start= time.time()
best_idx = classify(gt_image, scorer_parallel_jit, render, poses_to_score, h,w, object_indices)
end= time.time()
print ("Time elapsed:", end - start)
print(gt_mesh_name)
print(model_names[best_idx])



from IPython import embed; embed()
