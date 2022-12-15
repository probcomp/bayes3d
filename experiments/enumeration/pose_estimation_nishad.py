import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3.viz
from jax3dp3.rendering import render_planes
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos, depth_to_coords_in_camera
from jax.scipy.stats.multivariate_normal import logpdf
from jax.scipy.special import logsumexp
from jax3dp3.enumerations import fibonacci_sphere, geodesicHopf_select_axis
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.viz import save_depth_image
import jax3dp3.transforms_3d as t3d
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import jax3dp3.distributions
import matplotlib.pyplot as plt
import cv2
from jax3dp3.enumerations import make_translation_grid_enumeration, make_rotation_grid_enumeration

h, w = 200,200
fx,fy = 200, 200
cx,cy = 100, 100

r = 0.1
outlier_prob = 0.01

shapes = [get_rectangular_prism_shape(jnp.array([0.4, 0.4, 0.4])), get_rectangular_prism_shape(jnp.array([0.2, 0.2, 0.2]))]

render_from_pose = lambda pose, shape: render_planes(pose,shape, h,w,fx,fy,cx,cy)
render_from_pose_jit = jax.jit(render_from_pose)
render_from_pose_parallel = jax.vmap(render_from_pose, in_axes=(0,None))
render_from_pose_parallel_jit = jax.jit(render_from_pose_parallel)

center_of_sampling = t3d.transform_from_pos(jnp.array([0.0, 0.0, 3.0]))
variance = 0.5
concentration = 0.01
key = jax.random.PRNGKey(30)

sampler_jit = jax.jit(jax3dp3.distributions.gaussian_vmf_sample)

gt_pose = sampler_jit(key, center_of_sampling, variance, concentration)
gt_shape_idx = jax.random.categorical(key, jnp.ones(len(shapes))/len(shapes) )

gt_image = render_from_pose(gt_pose, shapes[gt_shape_idx])
save_depth_image(gt_image[:,:,2], "gt_image.png", max=5.0)

initial_pose_estimate = sampler_jit(jax.random.split(key)[0], center_of_sampling, variance, concentration)

translation_deltas_1 = make_translation_grid_enumeration(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 5, 5, 5)
translation_deltas_2 = make_translation_grid_enumeration(-0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 5, 5, 5)
translation_deltas_3 = make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5)
rotation_deltas = make_rotation_grid_enumeration(20, 20)

def scorer(pose, shape, gt_image):
    rendered_image = render_from_pose(pose, shape)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes = (0, None, None))

def coarse_to_fine_inference(x, shape, gt_image):
    proposals = jnp.einsum("ij,ajk->aik", x, translation_deltas_1)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", x, translation_deltas_2)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", x, translation_deltas_3)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
    weights_new = scorer_parallel(proposals, shape, gt_image)
    x = proposals[jnp.argmax(weights_new)]
    return x

coarse_to_fine_inference_jit = jax.jit(coarse_to_fine_inference)
pose_estimates_for_all_objects = [
    coarse_to_fine_inference_jit(initial_pose_estimate, shapes[i], gt_image)
    for i in range(len(shapes))
]
final_scores = [scorer(pose_estimates_for_all_objects[i], shapes[i], gt_image).item()  for i in range(len(shapes))]



# Each object panel
for i in range(len(shapes)):
    max_depth = 10.0
    gt_img_viz = jax3dp3.viz.get_depth_image(gt_image[:,:,2],max=max_depth) 
    initial_viz = jax3dp3.viz.get_depth_image(render_from_pose(initial_pose_estimate, shapes[i])[:,:,2],max=max_depth) 
    final_viz = jax3dp3.viz.get_depth_image(render_from_pose(pose_estimates_for_all_objects[i], shapes[i])[:,:,2],max=max_depth) 

    jax3dp3.viz.multi_panel(
        [gt_img_viz, initial_viz, final_viz],
        ["Ground Truth", "Initial", "Final"],
        10,
        50,
        20
    ).save("pose_estimation_{}.png".format(i))

# All objects panel

gt_img_viz = jax3dp3.viz.get_depth_image(gt_image[:,:,2],max=max_depth) 
final_imgs = []
for i in range(len(shapes)):
    final_viz = jax3dp3.viz.get_depth_image(render_from_pose(pose_estimates_for_all_objects[i], shapes[i])[:,:,2],max=max_depth) 
    final_imgs.append(final_viz)

jax3dp3.viz.multi_panel(
    [gt_img_viz] + final_imgs,
    ["Ground Truth\nShape {}\nPred Shape {}".format(gt_shape_idx, np.argmax(final_scores))]+ ["Shape {:d}\n Score {:0.3f}".format(i, final_scores[i]) for i in range(len(shapes))],
    10,
    100,
    20
).save("all_estimates.png")

from IPython import embed; embed()

