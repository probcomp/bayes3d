import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax3dp3.enumerations import get_rotation_proposals, make_translation_grid_enumeration
from jax3dp3.rendering import render_planes
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.viz import multi_panel, save_depth_image, get_depth_image

import numpy as np

## Camera / render setup

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

# default pose
x0, y0, z0 = 0.0, 0.0, 1.50
default_pose = jnp.array([
    [1.0, 0.0, 0.0, x0],   
    [0.0, 1.0, 0.0, y0],   
    [0.0, 0.0, 1.0, z0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

# c2f setup
likelihood_r, grid_width, num_grid_points = (0.02, 0.1, 10)
fib_nums, rot_nums = (30, 30)

enumeration_grid_tr = make_translation_grid_enumeration(
                        -grid_width, -grid_width, -grid_width,
                        grid_width, grid_width, grid_width,
                        num_grid_points,num_grid_points,num_grid_points)
enumeration_grid_rot = get_rotation_proposals(fib_nums, rot_nums)


def round(float_num, places=3):
    return int(float_num * 10**places)/10**places


## load test shapes
shapes_dims = jnp.array([jnp.array([0.5, 0.5, 0.5]), jnp.array([0.45, 0.5, 0.55]), jnp.array([0.40, 0.42, 0.8]), jnp.array([0.3, 0.5, 0.7]), jnp.array([0.1, 0.5, 0.5])])
shapes = [get_rectangular_prism_shape(shape_dims) for shape_dims in shapes_dims]
shape_labels = [str(shape_dims) for shape_dims in shapes_dims]
num_shapes = len(shapes)

## load test results
gt_poses = jnp.load(open('gt_test_poses.npy', 'rb') )   # best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape
predictions = jnp.load(open('predictions_all.npy', 'rb') )   # best idx, best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape
num_tests = len(predictions)
# assert num_tests == len(gt_poses)


# viz settings
middle_width = 20
top_border = 50
num_images = 1 + num_shapes   

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

width_scaler = 2
height_scaler = 2


## Setup testing 

for TEST_IDX in range(num_tests):
    gt_shape_idx = TEST_IDX % num_shapes
    gt_pose = gt_poses[TEST_IDX]


    gt_image = render_planes(gt_pose, shapes[gt_shape_idx],h,w,fx_fy,cx_cy) 
    gt_depth_image = get_depth_image(gt_image[:,:,2], max=max_depth)
    # gt_depth_image.save(f"gt_img.png")  

    print(f"testing image #{TEST_IDX}, gt shape {gt_shape_idx}")
    print(f"gt_pose={gt_pose}")

    preds = predictions[TEST_IDX]

    images = [gt_depth_image]
    labels = [f"GT:{shape_labels[gt_shape_idx]}"]

    latent_pose = gt_pose
    for shape_i, shape in enumerate(shapes):
        # tr_idx, rot_idx, score = preds[num_shapes*0 + shape_i], preds[num_shapes*1 + shape_i], preds[num_shapes*2 + shape_i] 
        
        pred_pose = preds[shape_i*16:(shape_i+1)*16].reshape(4,4)
        score = preds[16*shape_i+shape_i]
        
        print("pred pose=", pred_pose)
        
        pred_image = render_planes(pred_pose, shapes[shape_i],h,w,fx_fy,cx_cy) 
        pred_depth_image = get_depth_image(pred_image[:,:,2], max=max_depth)
        images.append(pred_depth_image)
        labels.append(shape_labels[shape_i] + f"\nScore={round(score,4)}" )

    dst = multi_panel(images, labels, middle_width, top_border, 13)


    dst.save(f"{TEST_IDX}_best_img.png") 
from IPython import embed; embed()

