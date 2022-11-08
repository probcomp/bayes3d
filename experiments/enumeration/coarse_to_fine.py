import sys
import os
sys.path.append('.')

import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.icp import icp
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes, render_cloud_at_pose
from jax3dp3.enumerations import get_rotation_proposals, 
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.shape import get_cube_shape, get_corner_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points, depth_to_coords_in_camera, sample_cloud_within_r
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
from jax3dp3.viz.enum import enumeration_range_bbox_viz
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

outlier_prob = 0.01
pixel_smudge = 0
fx, fy = fx_fy
cx, cy = cx_cy   
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])


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


## Viz properties
middle_width = 20
top_border = 30
cm = plt.get_cmap("turbo")


### Define scoring functions
r = cube_length * 2
coarse_scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy ,r, outlier_prob)
coarse_scorer_parallel = jax.vmap(coarse_scorer, in_axes = (0, None))

fine_r = r/2
fine_scorer = make_scoring_function(shape, h, w, fx_fy, cx_cy, fine_r, outlier_prob)
fine_scorer_parallel = jax.vmap(fine_scorer, in_axes = (0, None))


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
inference_frames_coarse_jit = jax.jit(partial(inference_frames, coarse_scorer_parallel))
inference_frames_fine_jit = jax.jit(partial(inference_frames, fine_scorer_parallel))

inference_frames_coarse_jit(jnp.empty((1,100,100,4)), jnp.empty((4, 4, 4, 4)))  
inference_frames_coarse_jit(jnp.empty((1,100,100,4)), jnp.empty((4, 128, 4, 4))) 


inference_frames_fine_jit(jnp.empty((1,100,100,4)), jnp.empty((4, 2, 4, 4))) 
inference_frames_fine_jit(jnp.empty((1,100,100,4)), jnp.empty((4, 16, 4, 4))) 
inference_frames_fine_jit(jnp.empty((1,100,100,4)), jnp.empty((4, 64, 4, 4))) 


# -----------------------------------------------------------------------

C2F_ITERS = 3 

search_r = 4  # radius of search; start at entire world
search_stepsize = 0.5  # start at increments of 1


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
for stage in range(C2F_ITERS):
    if stage == 0:
        inference_frames_jit = inference_frames_coarse_jit
        likelihood_r = r
    else:
        inference_frames_jit = inference_frames_fine_jit
        likelihood_r = fine_r

    grid = make_centered_grid_enumeration_3d_points(search_r, search_r, search_r, tr_half_steps, tr_half_steps, tr_half_steps)
    translation_proposals = jnp.einsum("ij,ajk->aik", current_pose_center, f_jit(grid))
    rotation_proposals = get_rotation_proposals(fib_numsteps, rot_numsteps)  

    cx, cy, cz = current_pose_center[:3, -1]
    _ = enumeration_range_bbox_viz(get_depth_image(best_image[:,:,2], 5.0), cx-search_r, cx+search_r, \
                            cy-search_r, cy+search_r, \
                            cz-search_r, cz+search_r, \
                            fx_fy, cx_cy, f"{stage}_search_range.png")


    print(f"Search radius: {search_r} ; Stepsize:{search_r/tr_half_steps}; Num steps {tr_half_steps}")
    print("enumerating over ", rotation_proposals.shape[0], " rotations")
    print("enumerating over ", translation_proposals.shape[0], " translations")

    translation_proposals_batches = batch_split(translation_proposals, NUM_TRANSLATE_BATCHES)
    rotation_proposals_batches = batch_split(rotation_proposals, NUM_ROTATE_BATCHES)

    print("num transl proposals=", translation_proposals_batches.shape)
    print("num pose proposals=", rotation_proposals_batches.shape)

    elapsed = 0
    _ = inference_frames_jit(test_gt_images, translation_proposals_batches)  
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
    save_depth_image(best_image[:,:,2], 5.0, f"{stage}_joint_img_fast.png")
    total_inf_time += elapsed

    print("\n Viz ... \n")
    all_images = []

    gt_depth_img = get_depth_image(observed[:,:,2], 5.0)
    pose_hypothesis_depth = render_planes_lambda(current_pose_center)[:, :, 2] #sample_likelihood this
    cloud = depth_to_coords_in_camera(pose_hypothesis_depth, K)[0]
    sampled_cloud_r = sample_cloud_within_r(cloud, r)  # new cloud
    rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx_fy, cx_cy, pixel_smudge)

    hypothesis_img = get_depth_image(rendered_cloud_r[:, :, 2], 5.0)
    for i in range(translation_proposals.shape[0]//2): #NUMBER OF TRANSLATION PROPOSALS TO VIZ):
        transl_proposal_image = get_depth_image(render_planes_lambda(translation_proposals[2*i])[:, :, 2], 5.0)
        images = [gt_depth_img, hypothesis_img, transl_proposal_image]
        labels = ["GT Image", f"Likelihood evaluation\nr={likelihood_r}", f"Enumeration\ngridscale={search_r}"]
        dst = multi_panel(images, labels, middle_width, top_border, 8)
        all_images.append(dst)
    
    all_images[0].save(
        fp=f"{stage}_out.gif",
        format="GIF",
        append_images=all_images,
        save_all=True,
        duration=50,
        loop=0,
    )


    # narrow search space depending on best pose proposal
    search_r /= tr_half_steps  # new search space has size equal to current search_stepsize
    tr_half_steps = max(2, int(tr_half_steps/2)) # reduce number of steps in new search space
    fib_numsteps, rot_numsteps = fib_numsteps*2, rot_numsteps*2
    current_pose_center = f(jnp.array([best_pose_proposal[0, :3, -1]]))[0]


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
save_depth_image(icp_img[:,:,2], 5.0, f"{stage}_icp_img_fast.png")




# #### Viz


# max_depth = 30.0
# middle_width = 20
# top_border = 100
# cm = plt.get_cmap("turbo")
# all_images = []
# for i in range(NUMBER OF TRANSLATION PROPOSALS TO VIZ):

#     gt_img_viz = same image as always

#     r_img_viz = sample(r) at current best pose

#     enums_viz = render(at best pose)

#     images = [gt_img_viz, r_img_viz, enums_viz]
#     labels = ["GT Image", "Depth Image", "Enumeration"]
#     dst = multi_panel(images, labels, middle_width, top_border, 40)
#     all_images.append(dst)








#     rgb = rgb_imgs[i]
#     rgb_img = Image.fromarray(
#         rgb.astype(np.int8), mode="RGBA"
#     )

#     depth_img = Image.fromarray(
#         np.rint(
#             cm(np.array(ground_truth_images[i, :, :, 2]) / max_depth) * 255.0
#         ).astype(np.int8),
#         mode="RGBA",
#     ).resize((original_width,original_height))

#     pose = x[i,-1,:,:]
#     rendered_image = render_from_pose_jit(pose)
#     rendered_depth_img = Image.fromarray(
#         (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
#     ).resize((original_width,original_height))

#     i1 = rendered_depth_img.copy()
#     i2 = rgb_img.copy()
#     i1.putalpha(128)
#     i2.putalpha(128)
#     overlay_img = Image.alpha_composite(i1, i2)

#     images = [rgb_img, depth_img, rendered_depth_img, overlay_img]
#     labels = ["RGB Image", "Depth Image", "Inferred Depth", "Overlay"]
#     dst = multi_panel(images, labels, middle_width, top_border, 40)
#     all_images.append(dst)


# all_images[0].save(
#     fp="out.gif",
#     format="GIF",
#     append_images=all_images,
#     save_all=True,
#     duration=100,
#     loop=0,
# )


from IPython import embed; embed()
