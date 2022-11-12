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
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.shape import get_cube_shape, get_corner_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points, depth_to_coords_in_camera
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
from jax3dp3.bbox import overlay_bounding_box
import time
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
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
    [1.0, 0.0, 0.0, gx],   
    [0.0, 1.0, 0.0, gy],   
    [0.0, 0.0, 1.0, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)


cube_length = 0.5
shape = get_cube_shape(cube_length)
NUM_BATCHES = 8

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


batched_scorer_parallel_jit = jax.jit(partial(batched_scorer_parallel, scorer_parallel, NUM_BATCHES))

# start = time.time()
# _weight = batched_scorer_parallel_jit(jnp.array(np.random.rand(1000,4,4)), jnp.array(np.random.rand(*gt_image.shape)), 0.5)
# jnp.argmax(_weight)
# print(time.time() - start)

# start = time.time()
# _weight = batched_scorer_parallel_jit(jnp.array(np.random.rand(1000,4,4)), jnp.array(np.random.rand(*gt_image.shape)), 0.)
# jnp.argmax(_weight)
# print(time.time() - start)

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

# tuples of (radius, width of gridding, grid_resolution)
schedule_tr = [(0.5, 1, 10), (0.25, 0.5, 10), (0.1, 0.2, 10), (0.02, 0.1, 10)]
schedule_rot = [(10, 10), (20, 20), (30, 30), (30,30)]


TOTAL_INF_TIME = 0

cnt = 0

for (r, grid_width, num_grid_points), (fib_nums, rot_nums) in zip(schedule_tr, schedule_rot):
    all_images = []

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
        
    # weights = batched_scorer_parallel_jit(proposals, gt_image, r) 
    # best_pose_estimate = proposals[jnp.argmax(weights)]
    
    if cnt == 0: # first-time compile
        weights = batched_scorer_parallel_jit(jnp.zeros(proposals.shape), jnp.zeros(gt_image.shape), r) 
        best_pose_estimate = proposals[jnp.argmax(weights)]
    cnt += 1
    

    start = time.time()
    weights = batched_scorer_parallel_jit(proposals, gt_image, r) 
    best_pose_estimate = proposals[jnp.argmax(weights)]
    end = time.time()
    print("inf time=", end-start, ", FPS=", 1/(end-start))
    TOTAL_INF_TIME += (end - start)

    # # rotation
    # enumerations_r = get_rotation_proposals(fib_nums, rot_nums)  
    # proposals = jnp.einsum('ij,ajk->aik', best_pose_estimate, enumerations_r)
    # proposals_batches = batch_split(proposals, 4)
    # print("(rot) proposals batches size=", proposals_batches.shape)
    # weights = batched_scorer_parallel(scorer_parallel, proposals_batches, gt_image, r).reshape(-1)
    # best_pose_estimate = proposals[jnp.argmax(weights)]


    print('best_pose_estimate:');print(best_pose_estimate) 
    if VIZ_MODE:
        print("Viz...")

        pose_hypothesis_depth = render_planes_lambda(gt_pose)[:, :, 2] #sample_likelihood this
        cloud = depth_to_coords_in_camera(pose_hypothesis_depth, K)[0]
        sampled_cloud_r = sample_cloud_within_r(cloud, r)  # new cloud
        rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx_fy, cx_cy, pixel_smudge)

        hypothesis_depth_img = get_depth_image(rendered_cloud_r[:, :, 2], max_depth)


        viz_stepsize = viz_stepsizes.pop()
        latent_pose_x, latent_pose_y, latent_pose_z = latent_pose_estimate[:3, -1]
        for i in range(proposals.shape[0]//viz_stepsize): 
            transl_proposal_image = get_depth_image(render_planes_lambda(proposals[i*viz_stepsize])[:, :, 2], max_depth)
            
            
            transl_proposal_image = overlay_bounding_box(latent_pose_x, latent_pose_y, latent_pose_z, grid_width, transl_proposal_image, K, save=False)
        
            images = [img.resize((w*width_scaler, h*height_scaler)) for img in (gt_depth_img, hypothesis_depth_img, transl_proposal_image)]
            labels = [f"GT Image\n latent pose={gx, gy, gz}\n Rotx={eulerx}, Roty={eulery}, Rotz={eulerz}", \
            f"Likelihood evaluation\nr={r},\nlatent pose={int(latent_pose_x*100)/100, int(latent_pose_y*100)/100, int(latent_pose_z*100)/100}", \
            f"Enumeration\ngridscale={grid_width/num_grid_points}"]
            dst = multi_panel(images, labels, middle_width, top_border, 13)
            all_images.append(dst)
            print(i) 

        best_img = get_depth_image(render_planes_lambda(best_pose_estimate)[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))
        best_img.save(f"{cnt}_best_img.png")
        # viz: pause at best estimated pose
        pause_frames_at_best = 50
        best_pose_x, best_pose_y, best_pose_z = best_pose_estimate[:3, -1]
        for _ in range(pause_frames_at_best):
            transl_proposal_image = get_depth_image(render_planes_lambda(best_pose_estimate)[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))
            images = [img.resize((w*width_scaler, h*height_scaler)) for img in (gt_depth_img, hypothesis_depth_img, transl_proposal_image)]
            labels = [f"GT Image\n latent pose={gx, gy, gz}\n Rotx={eulerx}, Roty={eulery}, Rotz={eulerz}", \
            f"Likelihood evaluation\nr={r},\nlatent pose={int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z*100)/100}", \
            f"BEST ESTIMATED POSE=\n{int(best_pose_x*100)/100, int(best_pose_y*100)/100, int(best_pose_z*100)/100}"]
            dst = multi_panel(images, labels, middle_width, top_border, 13)
            all_images.append(dst)

    latent_pose_estimate = best_pose_estimate

    print("\n------------------\n")


    if VIZ_MODE:
        all_images[0].save(
            fp=f"coarse_to_fine_{cnt}.gif",
            format="GIF",
            append_images=all_images,
            save_all=True,
            duration=25,
            loop=0,
        )


# once pose inference tuned reasonably, add a final correction step based on icp...
icp_jit = jax.jit(partial(icp, render_planes_jit))
_ = icp_jit(best_pose_estimate, observed, 1, 1)
icp_start = time.time()
# new_pose = icp_jit(best_pose_estimate, observed, 20, 2)
icp_end = time.time()  
icp_elapsed = icp_end - icp_start 
TOTAL_INF_TIME += icp_elapsed
print("Time elapsed for icp:", icp_elapsed, "; total inference time:", TOTAL_INF_TIME)
print("Total FPS:", 1/ TOTAL_INF_TIME)
# icp_img = render_planes_jit(new_pose)
# save_depth_image(icp_img[:,:,2], max_depth, f"C2F_icp_img.png")


from IPython import embed; embed()
