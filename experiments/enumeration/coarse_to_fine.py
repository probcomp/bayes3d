import sys
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.icp import icp
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.likelihood import sample_cloud_within_r, threedp3_likelihood
from jax3dp3.rendering import render_planes, render_cloud_at_pose
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.shape import get_cube_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points, depth_to_coords_in_camera
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
from jax3dp3.bbox import overlay_bounding_box
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from functools import partial 
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.enumerations import make_translation_grid_enumeration
from jax3dp3.transforms_3d import transform_from_pos

## Camera settings
h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
pixel_smudge = 0
fx, fy = fx_fy
cx, cy = cx_cy   
max_depth = 5.0
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

## Viz properties
VIZ_MODE = False  
all_images = []
viz_stepsizes = [2,2,2,2]


middle_width = 20
top_border = 50
num_images = 2  # gt and smapled

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

width_scaler = 2
height_scaler = 2

### Generate GT images
gx, gy, gz = 0.531, 0.251, 1.950
eulerx, eulery, eulerz = 0, 0, 0
gt_pose = jnp.array([
    [0.9860675,  -0.16779144, -0.04418374, gx],   
    [0.17300624,  0.92314297,  0.33919233, gy],   
    [-0.01606147, -0.34134597,  0.94141835, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)


cube_length = 0.5
shape = get_cube_shape(cube_length)
NUM_BATCHES = 4

render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))
gt_image = render_planes_jit(gt_pose)


def scorer(pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes = (0, None, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

batched_scorer_parallel = partial(batched_scorer_parallel, scorer_parallel, NUM_BATCHES)
batched_scorer_parallel_jit = jax.jit(batched_scorer_parallel)

observed = gt_image
print("GT image shape=", observed.shape)
print("GT pose=", gt_pose)

gt_depth_img = get_depth_image(observed[:,:,2], max_depth)
save_depth_image(observed[:,:,2], max_depth, "gt_img.png")

# Example of nans
nans = scorer(transform_from_pos(jnp.array([0.0, 0.0, -10.0])), gt_image, 0.1)


latent_pose_estimate = jnp.array([
    [1.0, 0.0, 0.0, 0.5],   
    [0.0, 1.0, 0.0, 0.25],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ])
    
print("initial latent=", latent_pose_estimate)

# tuples of (radius, width of gridding, num gridpoints)
schedule_tr = [(0.5, 1, 10), (0.25, 0.5, 10), (0.1, 0.2, 10), (0.02, 0.1, 10)]
schedule_rot = [(10, 10), (10, 10), (20, 20), (30,30)]

enumeration_likelihood_r = [sched[0] for sched in schedule_tr]
enumeration_grid_tr = [make_translation_grid_enumeration(
                        -grid_width, -grid_width, -grid_width,
                        grid_width, grid_width, grid_width,
                        num_grid_points,num_grid_points,num_grid_points
                    ) for (_, grid_width, num_grid_points) in schedule_tr]
enumeration_grid_r = [get_rotation_proposals(fib_nums, rot_nums) for (fib_nums, rot_nums) in schedule_rot]

TOTAL_INF_TIME = 0


def coarse_to_fine(enum_grid_tr, enum_grid_r, enum_likelihood_r, latent_pose_estimate, gt_image):
    for cnt in range(len(enum_likelihood_r)):
        r = enum_likelihood_r[cnt]
        enumerations_t =  enum_grid_tr[cnt]
        proposals = jnp.einsum("...ij,...jk->...ik", latent_pose_estimate, enumerations_t)

        # translation inference
        weights = batched_scorer_parallel(proposals, gt_image, r) 
        best_pose_estimate = proposals[jnp.argmax(weights)]

        # rotation inference
        enumerations_r = enum_grid_r[cnt]
        proposals = jnp.einsum('ij,ajk->aik', best_pose_estimate, enumerations_r)

        weights = batched_scorer_parallel(proposals, gt_image, r) 
        best_pose_estimate = proposals[jnp.argmax(weights)]

        latent_pose_estimate = best_pose_estimate


    return best_pose_estimate


# coarse_to_fine = partial(coarse_to_fine, enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r)
coarse_to_fine_jit = jax.jit(coarse_to_fine)

_ = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r,
                        jnp.array([
                        [1.0, 0.0, 0.0, 0.0],   
                        [0.0, 1.0, 0.0, 0.0],   
                        [0.0, 0.0, 1.0, 0.0],   
                        [0.0, 0.0, 0.0, 0.0],   
                        ]), jnp.zeros(gt_image.shape))

start = time.time()                       
best_pose_estimate = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r, latent_pose_estimate, gt_image)             
# best_pose_estimate = coarse_to_fine_jit(latent_pose_estimate, gt_image)
end = time.time()
print(best_pose_estimate)
elapsed = end - start
print("time elapsed = ", elapsed, " FPS=", 1/elapsed)
best_img = render_planes_jit(best_pose_estimate)
save_depth_image(best_img[:,:,2], max_depth, f"c2f_out.png")

from IPython import embed; embed()
