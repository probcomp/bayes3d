import sys
import os
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.icp import icp
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.shape import get_cube_shape, get_corner_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points
from jax3dp3.viz.img import save_depth_image
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from functools import partial 

OBJ_DEFAULT_POSE = TEMPLATE_DEFAULT_POSE = jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
) 

h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

r = 0.1
outlier_prob = 0.01
pixel_smudge = 0



### Generate GT images

num_frames = 50

gt_poses = [
    jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
]
rot = R.from_euler('zyx', [1.0, -0.1, -2.0], degrees=True).as_matrix()
delta_pose =     jnp.array([
    [1.0, 0.0, 0.0, 0.15],   
    [0.0, 1.0, 0.0, 0.05],   
    [0.0, 0.0, 1.0, 0.02],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
delta_pose = delta_pose.at[:3,:3].set(jnp.array(rot))

for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))
gt_poses = jnp.stack(gt_poses)
print("gt_poses.shape", gt_poses.shape)

cube_length = 0.5
shape = get_cube_shape(cube_length)

render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))
gt_images = render_planes_parallel_jit(gt_poses)



### Define scoring functions

scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy ,r, outlier_prob)
scorer_parallel = jax.vmap(scorer, in_axes = (0, None))


test_gt_images = []
test_frame_idx = 10

observed = gt_images[test_frame_idx, :, :, :]
print("GT image shape=", observed.shape)
print("GT pose=", gt_poses[test_frame_idx])

save_depth_image(observed[:,:,2], 5.0, "gt_img.png")

test_gt_images.append(observed)
test_gt_images = jnp.stack(test_gt_images)
observed_depth = observed[:, :, 2]

occ_x, occ_y, occ_z = observed[observed_depth > 0, :3].T
occ_xmin, occ_xmax, occ_ymin, occ_ymax, occ_zmin, occ_zmax = jnp.min(occ_x), jnp.max(occ_x), jnp.min(occ_y), jnp.max(occ_y), jnp.min(occ_z), jnp.max(occ_z)

# jitted helpers
f = (jax.vmap(lambda t: 
        jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]  # careful on whether it should be [0,0,0,1] when not joint
    )))
f_jit = jax.jit(f)
batch_split = lambda proposals, num_batches: jnp.array(jnp.split(proposals, num_batches))

############
# Single-frame coarse to fine
############



inference_frames = jax.vmap(enumerative_inference_single_frame, in_axes=(None, 0, None)) # vmap over images
inference_frames_jit = jax.jit(partial(inference_frames, scorer_parallel))

# -----------------------------------------------------------------------
# start at center w/o rotation
# GT: 
C2F_ITERS = 3 


# search space

search_r = 4  # radius of search; start at entire world, which we can say is 10x10x10
search_stepsize = 1  # start at increments of 1


current_pose_center = jnp.array([
    [1.0, 0.0, 0.0, 0],   
    [0.0, 1.0, 0.0, 0],   
    [0.0, 0.0, 1.0, 0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
) 

fib_numsteps = 6  # TODO rotation c2f
rot_numsteps = 4
tr_half_steps = 8 # translation # of steps per radius (TODO: must not exceed cube size)
NUM_TRANSLATE_BATCHES = 4  # TODO numbers
NUM_ROTATE_BATCHES = 4

for stage in range(C2F_ITERS):
    grid = make_centered_grid_enumeration_3d_points(search_r, search_r, search_r, tr_half_steps, tr_half_steps, tr_half_steps)
    translation_proposals = jnp.einsum("ij,ajk->aik", current_pose_center, f_jit(grid))
    rotation_proposals = get_rotation_proposals(fib_numsteps, rot_numsteps)  

    print(f"Current pose center: {current_pose_center}")
    print(f"Search radius: {search_r} ; Stepsize:{search_r/tr_half_steps}; Num steps {tr_half_steps}")
    print("enumerating over ", rotation_proposals.shape[0], " rotations")
    print("enumerating over ", translation_proposals.shape[0], " translations")

    translation_proposals_batches = batch_split(translation_proposals, NUM_TRANSLATE_BATCHES)
    rotation_proposals_batches = batch_split(rotation_proposals, NUM_ROTATE_BATCHES)

    elapsed = 0
    _ = inference_frames_jit(test_gt_images, translation_proposals_batches)  
    # 1) best translation
    start = time.time()
    best_translation_proposal, _ = inference_frames_jit(test_gt_images, translation_proposals_batches)  
    end = time.time()
    elapsed += (end-start)

    # 2) best pose at translation
    pose_proposals_batches = jnp.einsum('ij,abjk->abik', best_translation_proposal[0], rotation_proposals_batches)
    start = time.time()
    best_pose_proposal, _ = inference_frames_jit(test_gt_images, pose_proposals_batches)
    end = time.time()
    elapsed += (end-start)


    print("Time elapsed:", elapsed)
    print("FPS:", len(test_gt_images) / elapsed)
    print("Best pose=", best_pose_proposal)
    best_image = render_planes_lambda(best_pose_proposal[0])
    save_depth_image(best_image[:,:,2], 5.0, f"{stage}_joint_img.png")


    # narrow search space depending on best pose proposal
    search_r /= tr_half_steps  # new search space
    tr_half_steps = max(2, int(tr_half_steps * 0.5)) # reduce number of steps in the smaller space
    fib_numsteps, rot_numsteps = fib_numsteps*2, rot_numsteps*2
    current_pose_center = f(jnp.array([best_pose_proposal[0, :3, -1]]))[0]

from IPython import embed; embed()



# # ### TODO: once pose inference tuned reasonably, add a final correction step based on icp...
# icp_jit = jax.jit(partial(icp, render_planes_jit))
# _ = icp_jit(best_pose_proposal, observed, 1, 1)
# icp_start = time.time()
# new_pose = icp_jit(best_pose_proposal, observed, 20, 1)
# icp_end = time.time()  
# icp_elapsed = icp_end - icp_start 
# print("Time elapsed for icp:", icp_elapsed, "; total inference time:", icp_elapsed + elapsed)
# icp_img = render_planes_jit(new_pose)
# save_depth_image(icp_img[:,:,2], 5.0, f"{stage}_icp_img.png")