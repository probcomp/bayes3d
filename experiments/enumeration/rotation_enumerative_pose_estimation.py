import sys
import os
sys.path.append('.')


import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.viz.gif import make_gif
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos
from jax.scipy.stats.multivariate_normal import logpdf
from jax.scipy.special import logsumexp
from jax3dp3.enumerations import fibonacci_sphere, geodesicHopf_select_axis
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.viz import save_depth_image
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2

h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)
r = 0.1
outlier_prob = 0.01
pixel_smudge = 0

shape = get_rectangular_prism_shape(jnp.array([0.5, 0.4, 0.9]))
# shape = get_cube_shape(0.1)

render_from_pose = lambda pose, shape: render_planes(pose,shape,h,w,fx_fy,cx_cy)
render_from_pose_jit = jax.jit(render_from_pose)
render_from_pose_parallel = jax.vmap(render_from_pose, in_axes=(0,None))
render_from_pose_parallel_jit = jax.jit(render_from_pose_parallel)

gt_pose = jnp.array([
    [1.0, 0.0, 0.0, -0.0],   
    [0.0, 1.0, 0.0, -0.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [0.1, -0.3, -0.6], degrees=False).as_matrix()
gt_pose = gt_pose.at[:3,:3].set(jnp.array(rot))
gt_image = render_from_pose(gt_pose, shape)
save_depth_image(gt_image[:,:,2], 5.0, "gt_img.png")



num_sphere_gridpoints = 100
num_planar_angle_gridpoints = 24
unit_sphere_directions = fibonacci_sphere(num_sphere_gridpoints)
planar_rotations = jnp.linspace(0, 2*jnp.pi, num_planar_angle_gridpoints+1)[:-1]
geodesicHopf_select_axis_vmap = jax.vmap(jax.vmap(geodesicHopf_select_axis, in_axes=(0,None)), in_axes=(None,0))
rotation_enumerations = geodesicHopf_select_axis_vmap(unit_sphere_directions, planar_rotations).reshape(-1,4,4)
print("enumerating over ", rotation_enumerations.shape[0], " rotations")

original_translation = transform_from_pos(gt_pose[:3,-1])
potential_poses = jnp.einsum("ij,ajk->aik", original_translation, rotation_enumerations)

def scorer(pose, gt_image):
    rendered_image = render_from_pose(pose, shape)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes = (0, None))

def find_best_pose(initial_pose, gt_image):
    potential_poses = jnp.einsum("ij,ajk->aik", initial_pose, rotation_enumerations)
    weights_new = scorer_parallel(potential_poses, gt_image)
    x = potential_poses[jnp.argmax(weights_new)]
    return x
find_best_pose_jit = jax.jit(find_best_pose)


NUM_BATCHES = rotation_enumerations.shape[0] // 300   # anything greater than 300 likely leads to memory allocation err
assert rotation_enumerations.shape[0] % NUM_BATCHES == 0  # needs to be even split; see num_sphere_gridpoints, num_planar_angle_gridpoints
print(f"{NUM_BATCHES} batches")
def find_best_pose_over_batches(initial_pose, gt_image, num_batches): # scan over batches of rotation proposals for single image
    rotation_enumerations_batches = jnp.array(jnp.split(rotation_enumerations, NUM_BATCHES))
    def find_best_pose_in_batch(carry, rotation_enumerations_batch):
        # score over the selected rotation proposals
        proposals = jnp.einsum("ij,ajk->aik", initial_pose, rotation_enumerations_batch)  
        weights_new = scorer_parallel(proposals, gt_image)
        x, x_weight = proposals[jnp.argmax(weights_new)], jnp.max(weights_new)

        new_x, new_weight = jax.lax.cond(carry[-1] > jnp.max(weights_new), lambda: carry, lambda: (x, x_weight))

        return (new_x, new_weight), None  # return highest weight pose proposal encountered so far
    best_prop, _ = jax.lax.scan(find_best_pose_in_batch, (jnp.empty((4,4)), jnp.NINF), rotation_enumerations_batches)
    return best_prop[0]
find_best_pose_over_batches_jit = jax.jit(find_best_pose_over_batches)


_ = find_best_pose_over_batches_jit(original_translation,gt_image, NUM_BATCHES)  # 1st compile
start = time.time()
best_pose = find_best_pose_over_batches_jit(original_translation,gt_image, NUM_BATCHES)
end = time.time()
print ("Time elapsed:", end - start)
best_image = render_from_pose(best_pose, shape)
save_depth_image(best_image[:,:,2], 5.0, "img.png")
print('gt_pose:');print(gt_pose)
print('best_pose:');print(best_pose)




from IPython import embed; embed()
