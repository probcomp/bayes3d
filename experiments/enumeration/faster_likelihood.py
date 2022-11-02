import jax.numpy as jnp
import jax
import functools
from jax3dp3.utils import extract_2d_patches
import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import gaussian_vmf
from jax3dp3.viz.gif import make_gif 
from jax3dp3.likelihood import threedp3_likelihood, threedp3_likelihood_alternate
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
    101,
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

gt_pose = jnp.array(
    [
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

rendered_pose = jnp.array([
    [1.0, 0.0, 0.0, -0.0],   
    [0.0, 1.0, 0.0, -0.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rendered_image = render_from_pose(rendered_pose, shape)

obs_xyz = gt_image
rendered_xyz = rendered_image

obs_mask = obs_xyz[:,:,2] > 0.0
rendered_mask = rendered_xyz[:,:,2] > 0.0

def scorer(pose, gt_image):
    rendered_image = render_from_pose(pose, shape)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.jit(jax.vmap(scorer, in_axes = (0, None)))

def scorer_alternate(pose, gt_image):
    rendered_image = render_from_pose(pose, shape)
    weight = threedp3_likelihood_alternate(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel_alternate = jax.jit(jax.vmap(scorer_alternate, in_axes = (0, None)))

N = 20000
all_poses = jnp.stack([rendered_pose for _ in range(N)])

scores = scorer_parallel(all_poses, gt_image)

start = time.time()
scores = scorer_parallel(all_poses, gt_image)
print(scores.shape)
end = time.time()
print ("Time elapsed:", end - start)


# scores = scorer_parallel_alternate(all_poses, gt_image)

# start = time.time()
# scores = scorer_parallel_alternate(all_poses, gt_image)
# end = time.time()
# print ("Time elapsed:", end - start)