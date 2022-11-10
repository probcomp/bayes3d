import sys
import os
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.icp import icp
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.likelihood import sample_cloud_within_r, threedp3_likelihood
from jax3dp3.rendering import render_planes, render_cloud_at_pose
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame, batch_split
from jax3dp3.shape import get_cube_shape, get_corner_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points, depth_to_coords_in_camera
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
from jax3dp3.viz.enum import enumeration_range_bbox_viz
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from functools import partial 
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.enumerations import make_grid_enumeration, make_translation_grid_enumeration
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos

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
middle_width = 20
top_border = 50
num_images = 2  # gt and smapled

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

width_scaler = 2
height_scaler = 2

### Generate GT images
gx, gy, gz = 0.531, 0.501, 1.950
eulerx, eulery, eulerz = 0, 0, 0
gt_pose = jnp.array([
    [1.0, 0.0, 0.0, gx],   
    [0.0, 1.0, 0.0, gy],   
    [0.0, 0.0, 1.0, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)


cube_length = 0.5
shape = get_cube_shape(cube_length)

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
batched_scorer_parallel_jit = jax.jit(partial(batched_scorer_parallel, scorer_parallel))


observed = gt_image
print("GT image shape=", observed.shape)
print("GT pose=", gt_pose)

gt_depth_img = get_depth_image(observed[:,:,2], max_depth).resize((w*width_scaler, h*height_scaler))
save_depth_image(observed[:,:,2], max_depth, "gt_img.png")

# Example of nans
nans = scorer(transform_from_pos(jnp.array([0.0, 0.0, -10.0])), gt_image, 0.1)


latent_pose_estimate = jnp.array([
    [1.0, 0.0, 0.0, 0],   
    [0.0, 1.0, 0.0, 0],   
    [0.0, 0.0, 1.0, 2],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)


# tuples of (radius, width of gridding, grid_resolution)
schedule_tr = [(0.1, 3, 10), (0.1, 1, 10), (0.05, 0.24, 3), (0.01, 0.05, 3)]
schedule_rot = [(10, 10), (20, 20), (30, 30), (30,30)]
viz_stepsizes = [9,9,25,25]

all_images = []
for (r, grid_width, num_grid_points), (fib_nums, rot_nums) in zip(schedule_tr, schedule_rot):
    print(f"r={r}, grid_width={grid_width}, num_grid_points={num_grid_points}")
    assert grid_width*2/num_grid_points - cube_length <= r
    estx, esty, estz = latent_pose_estimate[:3, -1]  # hacky but this is for bbox viz

    # translation
    enumerations_t = make_translation_grid_enumeration(
        -grid_width, -grid_width, -grid_width,
        grid_width, grid_width, grid_width,
        num_grid_points,num_grid_points,num_grid_points
    )
    proposals = jnp.einsum("...ij,...jk->...ik", latent_pose_estimate, enumerations_t)
    proposals_batches = batch_split(proposals, 4)
    print("(tr) proposals batches size=", proposals_batches.shape)
    weights = batched_scorer_parallel_jit(proposals_batches, gt_image, r).reshape(-1)
    best_pose_estimate = proposals[jnp.argmax(weights)]

    # # rotation
    # enumerations_r = get_rotation_proposals(fib_nums, rot_nums)  
    # proposals = jnp.einsum('ij,ajk->aik', current_pose_estimate, enumerations_r)
    # proposals_batches = batch_split(proposals, 4)
    # print("(rot) proposals batches size=", proposals_batches.shape)
    # weights = batched_scorer_parallel(scorer_parallel, proposals_batches, gt_image, r).reshape(-1)
    # current_pose_estimate = proposals[jnp.argmax(weights)]


    print('best_pose_estimate:');print(best_pose_estimate)


    # Viz
    print("Viz...")

    pose_hypothesis_depth = render_planes_lambda(latent_pose_estimate)[:, :, 2] #sample_likelihood this
    cloud = depth_to_coords_in_camera(pose_hypothesis_depth, K)[0]
    sampled_cloud_r = sample_cloud_within_r(cloud, r)  # new cloud
    rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx_fy, cx_cy, pixel_smudge)

    hypothesis_depth_img = get_depth_image(rendered_cloud_r[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))    

    # hypothesis_depth_img = enumeration_range_bbox_viz(hypothesis_depth_img, \
    #                     estx-grid_width, estx+grid_width, \
    #                     esty-grid_width, esty+grid_width, \
    #                     estz-grid_width, estz+grid_width, \
    #                     fx_fy, cx_cy, f"search_range.png")

    viz_stepsize = viz_stepsizes.pop()
    latent_pose_x, latent_pose_y, latent_pose_z = latent_pose_estimate[:3, -1]
    for i in range(proposals.shape[0]//viz_stepsize): 
        transl_proposal_image = get_depth_image(render_planes_lambda(proposals[i*viz_stepsize])[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))
        images = [gt_depth_img, hypothesis_depth_img, transl_proposal_image]
        labels = [f"GT Image\n latent pose={gx, gy, gz}\n rotx={eulerx}, roty={eulery}, rotz={eulerz}", \
        f"Likelihood evaluation\nr={r},\nlatent pose={int(latent_pose_x*100)/100, int(latent_pose_y*100)/100, int(latent_pose_z*100)/100}", \
        f"Enumeration\ngridscale={grid_width/num_grid_points}"]
        dst = multi_panel(images, labels, middle_width, top_border, 13)
        all_images.append(dst)


    # viz: pause at best estimated pose
    pause_frames_at_best = 100
    best_pose_x, best_pose_y, best_pose_z = best_pose_estimate[:3, -1]
    for _ in range(pause_frames_at_best):
        transl_proposal_image = get_depth_image(render_planes_lambda(best_pose_estimate)[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))
        images = [gt_depth_img, hypothesis_depth_img, transl_proposal_image]
        labels = [f"GT Image\n latent pose={gx, gy, gz}\n rotx={eulerx}, roty={eulery}, rotz={eulerz}", \
        f"Likelihood evaluation\nr={r},\nlatent pose={int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z*100)/100}", \
        f"BEST ESTIMATED POSE=\n{int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z*100)/100}"]
        dst = multi_panel(images, labels, middle_width, top_border, 13)
        all_images.append(dst)

    latent_pose_estimate = best_pose_estimate

    print("------------------")


all_images[0].save(
    fp=f"likelihood_out.gif",
    format="GIF",
    append_images=all_images,
    save_all=True,
    duration=75,
    loop=0,
)

from IPython import embed; embed()


############################# NISHADS STUFF ^ ############################# 



# -----------------------------------------------------------------------

C2F_ITERS = 3 

search_r = 4  # radius of search; start at entire world
search_stepsize = 0.25  # start at increments of 1


current_pose_center = jnp.array([
    [1.0, 0.0, 0.0, 0],   
    [0.0, 1.0, 0.0, 0],   
    [0.0, 0.0, 1.0, 0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
) 

fib_numsteps = 4  # TODO rotation c2f
rot_numsteps = 4
tr_half_steps = int(search_r/search_stepsize) # translation # of steps per radius (TODO: must not exceed cube size)

assert search_stepsize - cube_length <= r

NUM_TRANSLATE_BATCHES = 4  # TODO numbers
NUM_ROTATE_BATCHES = 4

total_inf_time = 0

best_image = jnp.zeros((100,100,4))
gt_depth_img = get_depth_image(observed[:,:,2], max_depth)
gt_pose_x, gt_pose_y, gt_pose_z = gt_pose[:3, -1]
gt_pose_x, gt_pose_y, gt_pose_z = int(gt_pose_x*100)/100, int(gt_pose_y*100)/100, int(gt_pose_z)*100/100

best_pose_x, best_pose_y, best_pose_z = current_pose_center[:3, -1]
best_pose_x, best_pose_y, best_pose_z = int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z)*100/100  # for viz


for stage in range(C2F_ITERS):
    if stage == 0:
        inference_frames_jit = inference_frames_coarse_jit
        likelihood_r = r
    elif stage == 1:
        inference_frames_jit = inference_frames_fine_jit
        likelihood_r = fine_r
    else:
        inference_frames_jit = inference_frames_finest_jit
        likelihood_r = finest_r


    grid = make_centered_grid_enumeration_3d_points(search_r, search_r, search_r, tr_half_steps, tr_half_steps, tr_half_steps)
    translation_proposals = jnp.einsum("ij,ajk->aik", current_pose_center, f_jit(grid))
    rotation_proposals = get_rotation_proposals(fib_numsteps, rot_numsteps)  

    cx, cy, cz = current_pose_center[:3, -1]
    bbox_viz = enumeration_range_bbox_viz(get_depth_image(best_image[:,:,2], max_depth), cx-search_r, cx+search_r, \
                            cy-search_r, cy+search_r, \
                            cz-search_r, cz+search_r, \
                            fx_fy, cx_cy, f"{stage}_search_range.png")


    print(f"Likelihood r:{likelihood_r}, Search radius: {search_r} ; Stepsize:{search_r/tr_half_steps}; Num steps {tr_half_steps}")
    print("enumerating over ", rotation_proposals.shape[0], " rotations")
    print("enumerating over ", translation_proposals.shape[0], " translations")

    translation_proposals_batches = batch_split(translation_proposals, NUM_TRANSLATE_BATCHES)
    rotation_proposals_batches = batch_split(rotation_proposals, NUM_ROTATE_BATCHES)

    print("num transl proposals=", translation_proposals_batches.shape)
    print("num pose proposals=", rotation_proposals_batches.shape)

    elapsed = 0
    # _ = inference_frames_jit(test_gt_images, translation_proposals_batches)  
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
    print("Best pose=", best_pose_proposal)
    best_image = render_planes_lambda(best_pose_proposal[0])
    save_depth_image(best_image[:,:,2], max_depth, f"{stage}_joint_img_fast.png")
    total_inf_time += elapsed

    print("\n Viz ... \n")
    all_images = []

    pose_hypothesis_depth = render_planes_lambda(current_pose_center)[:, :, 2] #sample_likelihood this
    cloud = depth_to_coords_in_camera(pose_hypothesis_depth, K)[0]
    sampled_cloud_r = sample_cloud_within_r(cloud, likelihood_r)  # new cloud
    rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx_fy, cx_cy, pixel_smudge)

    hypothesis_img = get_depth_image(rendered_cloud_r[:, :, 2], max_depth)
    # joint_proposals = jnp.einsum("aij,bjk->abik", translation_proposals, rotation_proposals).reshape(-1, 4, 4)
    # for i in range(translation_proposals.shape[0]//2): #NUMBER OF TRANSLATION PROPOSALS TO VIZ):
    #     transl_proposal_image = get_depth_image(render_planes_lambda(translation_proposals[2*i])[:, :, 2], max_depth)
    #     images = [gt_depth_img, hypothesis_img, transl_proposal_image]
    #     labels = [f"GT Image\n{gt_pose_x, gt_pose_y, gt_pose_z}", f"Likelihood evaluation\nr={likelihood_r},latent pose={best_pose_x, best_pose_y, best_pose_z}", f"Enumeration\ngridscale={search_r}"]
    #     dst = multi_panel(images, labels, middle_width, top_border, 8)
    #     all_images.append(dst)



    # all_images[0].save(
    #     fp=f"{stage}_out.gif",
    #     format="GIF",
    #     append_images=all_images,
    #     save_all=True,
    #     duration=50,
    #     loop=0,
    # )


    # narrow search space depending on best pose proposal
    search_r /= tr_half_steps  # new search space has size equal to current search_stepsize
    tr_half_steps = max(2, int(tr_half_steps/2)) # reduce number of steps in new search space
    fib_numsteps, rot_numsteps = fib_numsteps*2, rot_numsteps*2
    current_pose_center = f(jnp.array([best_pose_proposal[0, :3, -1]]))[0]
    best_pose_x, best_pose_y, best_pose_z = current_pose_center[:3, -1]
    best_pose_x, best_pose_y, best_pose_z = int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z)*100/100  # for viz rounding

    print("==================================")



# once pose inference tuned reasonably, add a final correction step based on icp...
icp_jit = jax.jit(partial(icp, render_planes_jit))
_ = icp_jit(best_pose_proposal[0], observed, 1, 1)
icp_start = time.time()
new_pose = icp_jit(best_pose_proposal[0], observed, 20, 2)
icp_end = time.time()  
icp_elapsed = icp_end - icp_start 
print("Time elapsed for icp:", icp_elapsed, "; total inference time:", icp_elapsed + total_inf_time)
print("Total FPS:", len(test_gt_images)/(icp_elapsed + total_inf_time))
icp_img = render_planes_jit(new_pose)
save_depth_image(icp_img[:,:,2], max_depth, f"{stage}_icp_img_fast.png")


