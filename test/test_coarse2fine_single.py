import sys
sys.path.append('.')

import jax.numpy as jnp
import jax
from jax3dp3.coarse_to_fine import coarse_to_fine, coarse_to_fine_pose_and_weights
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import time
from functools import partial 
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.enumerations import make_translation_grid_enumeration

## Camera settings
h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
fx, fy = fx_fy
cx, cy = cx_cy   
max_depth = 5.0
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

### Generate GT images
gx, gy, gz = 0.531, 0.251, 1.950
gt_pose = jnp.array([
    [0.9860675,  -0.16779144, -0.04418374, gx],   
    [0.17300624,  0.92314297,  0.33919233, gy],   
    [-0.01606147, -0.34134597,  0.94141835, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
gt_dims = jnp.array([0.5, 0.5, 0.5])
gt_shape = get_rectangular_prism_shape(gt_dims)
gt_image = render_planes(gt_pose, gt_shape, h, w, fx, fy, cx, cy)


# generate shapes to eval
shapes_dims = jnp.array([jnp.array([0.3, 0.5, 0.7]), jnp.array([0.5, 0.5, 0.5]), jnp.array([0.1, 0.5, 0.5])])
num_shapes = len(shapes_dims)
shapes_idxs = jnp.arange(num_shapes)


def scorer(shape, pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx, fy, cx, cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight

print("GT image shape=", gt_image.shape)
print("GT pose=", gt_pose)

gt_depth_img = get_depth_image(gt_image[:,:,2], max_depth)
save_depth_image(gt_image[:,:,2], "gt_img.png", max=max_depth)

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


def evaluate_shapes():
    coarse_to_fine_shape = lambda shape_id: coarse_to_fine_pose_and_weights(partial(scorer, get_rectangular_prism_shape(shapes_dims[shape_id])), 
                                                    enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r, 
                                                    latent_pose_estimate, gt_image)
    
    best_pose_estimates, best_pose_scores = jax.vmap(coarse_to_fine_shape, in_axes=(0,))(shapes_idxs)
    return best_pose_estimates, best_pose_scores

best_pose_estimates, best_pose_scores = evaluate_shapes()


## Viz setup
middle_width = 20
top_border = 50
num_images = 1 + num_shapes   

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

def round_str(float_num, places=3):
    return int(float_num * 10**places)/10**places

## Viz result
images = [get_depth_image(gt_image[:,:,2], max=max_depth)]
labels = [f"GT:{str(gt_dims)}"]

for shape_i in range(num_shapes):  
    pred_pose = best_pose_estimates[shape_i]
    score = best_pose_scores[shape_i]
    shape_dims = shapes_dims[shape_i]
    
    print("pred pose=", pred_pose)

    pred_image = render_planes(pred_pose, get_rectangular_prism_shape(shape_dims),h,w,fx,fy,cx,cy) 
    pred_depth_image = get_depth_image(pred_image[:,:,2], max=max_depth)


    images.append(pred_depth_image)
    labels.append(str(shape_dims) + f"\nScore={round_str(score,4)}" )

dst = multi_panel(images, labels, middle_width, top_border, 13)

dst.save(f"test_best_img.png") 

from IPython import embed; embed()

