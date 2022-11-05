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

### Setup

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

render_planes_jit = jax.jit(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy))
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))
gt_images = render_planes_parallel_jit(gt_poses)



### Define scoring functions

scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy ,r, outlier_prob)
scorer_parallel = jax.vmap(scorer, in_axes = (0, None))


test_gt_images = []
test_frame_idx = 10

observed = gt_images[test_frame_idx, :, :, :]

save_depth_image(observed[:,:,2], 5.0, "gt_img.png")

test_gt_images.append(observed)
test_gt_images = jnp.stack(test_gt_images)
observed_viz = (np.array(observed) / 10 * 255.0).astype('uint8')
observed_depth = observed[:, :, 2]
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints / descriptors with ORB
kp, des = orb.detectAndCompute(observed_viz,None)
observed_depth_kp = cv2.drawKeypoints(observed_viz[:, :, 2], kp, None, color=(0,255,0), flags=0)  # just for viz
# plt.imshow(observed_depth_kp), plt.show()

if len(kp) == 0:
    gx, gy, gz = -1.0, -1.0, 2.0
else:
    sample_keypoint = kp[0].pt  # take a sample keypoint to location match
    t_col, t_row = sample_keypoint
    t_col, t_row = int(t_col), int(t_row); print(t_col, t_row)
    gx, gy, gz = observed[t_row, t_col, :3]

translation_proposal = jnp.array([
    [1.0, 0.0, 0.0, gx],   
    [0.0, 1.0, 0.0, gy],   
    [0.0, 0.0, 1.0, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)  
print("translation proposal=", translation_proposal)


f_jit = jax.jit(jax.vmap(lambda t: 
        jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]  # careful on whether it should be [0,0,0,1] when not joint
        )))
translation_proposals = jnp.einsum("ij,ajk->aik", translation_proposal, f_jit(make_centered_grid_enumeration_3d_points(cube_length/3, cube_length/3, cube_length/3, 4, 4, 4)))


### Enumerative inference over rotation proposals
rotation_proposals = get_rotation_proposals(75, 20)
print("enumerating over ", rotation_proposals.shape[0], " rotations")
print("enumerating over ", translation_proposals.shape[0], " translations")
joint_proposals = jnp.einsum("aij,bjk->abik", translation_proposals, rotation_proposals).reshape(-1, 4, 4)
print("joint proposals size= ", joint_proposals.shape)


NUM_BATCHES_T = max(2, translation_proposals.shape[0] // 100)   # anything greater than 300 leads to memory allocation err
NUM_BATCHES_R = max(2, rotation_proposals.shape[0] // 100)   # anything greater than 300 leads to memory allocation err
translation_proposals_batches = jnp.array(jnp.split(translation_proposals, NUM_BATCHES_T))
rotation_proposals_batches = jnp.array(jnp.split(rotation_proposals, NUM_BATCHES_R))


inference_frames = jax.vmap(enumerative_inference_single_frame, in_axes=(None, 0, None)) # vmap over images
inference_frames_jit = jax.jit(partial(inference_frames, scorer_parallel))


# compile
inference_frames_jit(jnp.empty(test_gt_images.shape), jnp.empty(translation_proposals_batches.shape))  # a (num_images x 4 x 4) arr of best pose estim for each image frame
inference_frames_jit(jnp.empty(test_gt_images.shape), jnp.empty(rotation_proposals_batches.shape))  # time


elapsed = 0
# 1) best translation
start = time.time()
best_translation_proposal, _ = inference_frames_jit(test_gt_images, translation_proposals_batches)  
end = time.time()
elapsed += (end-start)

# 2) best pose at translation
start = time.time()
pose_proposals_batches = jnp.einsum('ij,abjk->abik', best_translation_proposal[0], rotation_proposals_batches)
best_pose_proposal, _ = inference_frames_jit(test_gt_images, pose_proposals_batches)
end = time.time()
elapsed += (end-start)

print("Time elapsed:", elapsed)
print("FPS:", len(test_gt_images) / elapsed)
print("best poses:", best_pose_proposal)


# ### TODO: once pose inference tuned reasonably, add a final correction step based on icp...
icp_jit = jax.jit(partial(icp, render_planes_jit))
_ = icp_jit(best_pose_proposal[0], observed, 1, 1)
icp_start = time.time()
new_pose = icp_jit(best_pose_proposal[0], observed, 20, 1)
icp_end = time.time()  
icp_elapsed = icp_end - icp_start 
print("Time elapsed for icp:", icp_elapsed, "; total inference time:", icp_elapsed + elapsed)
print("Overall FPS=", len(test_gt_images)/(icp_elapsed + elapsed))
icp_img = render_planes_jit(new_pose)
save_depth_image(icp_img[:,:,2], 5.0, "keypoint_icp_img.png")


best_image = render_planes_jit(best_pose_proposal[0])
save_depth_image(best_image[:,:,2], 5.0, "keypoint_joint_img.png")


from IPython import embed; embed()