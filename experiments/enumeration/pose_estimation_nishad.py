import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
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
from jax3dp3.viz.img import save_depth_image
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



num_sphere_gridpoints = 200
num_planar_angle_gridpoints = 75
unit_sphere_directions = fibonacci_sphere(num_sphere_gridpoints)
planar_rotations = jnp.linspace(0, 2*jnp.pi, num_planar_angle_gridpoints+1)[:-1]
geodesicHopf_select_axis_vmap = jax.vmap(jax.vmap(geodesicHopf_select_axis, in_axes=(0,None)), in_axes=(None,0))
rotation_enumerations = geodesicHopf_select_axis_vmap(unit_sphere_directions, planar_rotations).reshape(-1,4,4)

original_translation = transform_from_pos(gt_pose[:3,-1])
potential_poses = jnp.einsum("ij,ajk->aik", original_translation, rotation_enumerations)
print('potential_poses.shape:');print(potential_poses.shape)

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

best_pose = find_best_pose_jit(original_translation,gt_image)

start = time.time()
best_pose = find_best_pose_jit(original_translation,gt_image)
end = time.time()
print ("Time elapsed:", end - start)

best_image = render_from_pose(best_pose, shape)
save_depth_image(best_image[:,:,2], 5.0, "img.png")
print('gt_pose:');print(gt_pose)
print('best_pose:');print(best_pose)




from IPython import embed; embed()