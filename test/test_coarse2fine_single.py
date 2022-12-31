import sys
sys.path.append('.')

from functools import partial 
import numpy as np
import jax
from jax.experimental import host_callback
import jax.numpy as jnp
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.coarse_to_fine import coarse_to_fine, coarse_to_fine_pose_and_weights
from jax3dp3.enumerations import get_rotation_proposals, make_translation_grid_enumeration
from jax3dp3.metrics import get_rot_error_from_poses, get_translation_error_from_poses
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import time

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


## Scorer generations

def scorer(shape, pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx,fy,cx,cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight


#######################
# Run coarse-to-fine on test images
# Retrieve highest-likelihood proposal for each possible shape,
# and get the best hypothesis shape
#####################

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

# TODO: also make "schedule" for fine-only comparison

"""
Setup testing (see test/test_generate_enumerations.py)

Test Shapes:
Rectangular prism 0: (cube)
Rectangular prism 1: (near-cube)
Rectangular prism 2: (2 similar dims, 1 unsimilar dim)
Rectangular prism 3: (3 unsimilar dims)
Rectangular prism 4: (near-(square)plane)

Test GT Poses:
4x4x4 translation enumerations
8x8 rotation enumerations
(4096 pose proposals to test)
"""

# generate test shapes
x0, y0, z0 = 0.0, 0.0, 1.50
default_pose = jnp.array([
    [1.0, 0.0, 0.0, x0],   
    [0.0, 1.0, 0.0, y0],   
    [0.0, 0.0, 1.0, z0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
transl_noise = jnp.array([
    [0.0, 0.0, 0.0, 1e-3],   
    [0.0, 0.0, 0.0, -1e-4],   
    [0.0, 0.0, 0.0, 7e-5],   
    [0.0, 0.0, 0.0, 0.0],   
    ]
)

shapes_dims = jnp.array([jnp.array([0.1, 0.5, 0.5]), jnp.array([0.3, 0.5, 0.7]), jnp.array([0.5, 0.5, 0.5])]) #jnp.array([jnp.array([0.5, 0.5, 0.5]), jnp.array([0.45, 0.5, 0.55]), jnp.array([0.40, 0.42, 0.8]), jnp.array([0.3, 0.5, 0.7]), jnp.array([0.1, 0.5, 0.5])])
shapes = [get_rectangular_prism_shape(shape_dims) for shape_dims in shapes_dims]
shape_labels = [str(shape_dims) for shape_dims in shapes_dims]
num_shapes = len(shapes_dims)
all_shape_idxs = jnp.array([0,0,0])#jnp.arange(num_shapes)
print(f"Testing {num_shapes} shapes")

# generate test poses and corresponding gt shapes (the ith pose has (i%num_shapes)th shape)
# f_jit = jax.jit(jax.vmap(lambda t: jnp.vstack(
#                 [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]))
#              )
# pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.4, 0.4, 0.5, 3, 3, 3))
# translations = jnp.einsum("ij,ajk->aik", default_pose, pose_deltas)
# rotations = get_rotation_proposals(8,8)
# gt_test_poses = jnp.einsum("aij,bjk->abik", translations, rotations).reshape(-1, 4,4) 

gx, gy, gz = 0.531, 0.251, 1.950
gt_test_poses = jnp.array([[
    [0.9860675,  -0.16779144, -0.04418374, gx],   
    [0.17300624,  0.92314297,  0.33919233, gy],   
    [-0.01606147, -0.34134597,  0.94141835, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]]
)

print(f"Testing {len(gt_test_poses)} images (GT Poses)")

# generate test images
def render_test_images(gt_test_poses):
    images = []
    for test_idx, gt_test_pose in enumerate(gt_test_poses):
        gt_shape_id = test_idx % num_shapes
        gt_test_image = render_planes(gt_test_pose,shapes[gt_shape_id],h,w,fx,fy,cx,cy) 
        images.append(gt_test_image)
    
    return jnp.array(images)
gt_test_images = render_test_images(gt_test_poses)
print(f"Rendered {len(gt_test_images)} gt test images")

def get_c2f_func(shape_id):
    scorer_for_shape = partial(scorer, get_rectangular_prism_shape(shapes_dims[shape_id]))
    coarse_to_fine_func = partial(coarse_to_fine_pose_and_weights, scorer_for_shape, enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r)
    return coarse_to_fine_func


def get_prediction(test_idx):
    gt_test_image = gt_test_images[test_idx]
    latent_pose_estimate = jnp.array([
                            [1.0, 0.0, 0.0, 0.5],   
                            [0.0, 1.0, 0.0, 0.25],   
                            [0.0, 0.0, 1.0, 2.0],   
                            [0.0, 0.0, 0.0, 1.0],   
                            ])  

    # retrieve best pose hypothesis for each possible shape
    pose_estimator = lambda shape_idx: get_c2f_func(shape_idx)(latent_pose_estimate, gt_test_image) 
    best_pose_for_shape, best_pose_score_for_shape = jax.vmap(pose_estimator, in_axes=(0,))(all_shape_idxs)  

    best_shape_idx = jnp.argmax(best_pose_score_for_shape)  # get highest-score shape

    return best_pose_for_shape, best_pose_score_for_shape, best_shape_idx
get_prediction_jit = jax.jit(get_prediction)


test_idx = 0
gt_shape_id = test_idx % num_shapes
pred = get_prediction_jit(test_idx)
best_pose_for_shape, best_pose_score_for_shape, best_shape_idx = pred

from IPython import embed; embed()



## Viz setup
middle_width = 20
top_border = 50
num_images = 1 + num_shapes   

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

def round_str(float_num, places=3):
    return int(float_num * 10**places)/10**places

## Viz result
images = [get_depth_image(gt_test_images[test_idx][:,:,2], max=max_depth)]
labels = [f"GT:{shape_labels[gt_shape_id]}"]
gt_test_poses[test_idx], gt_test_images[test_idx]

print("GT pose=", gt_test_poses[test_idx])
for shape_i, shape in enumerate(shapes):  
    pred_pose = best_pose_for_shape[shape_i]
    score = best_pose_score_for_shape[shape_i]
    
    print("pred pose=", pred_pose)

    pred_image = render_planes(pred_pose, shapes[shape_i],h,w,fx,fy,cx,cy) 
    pred_depth_image = get_depth_image(pred_image[:,:,2], max=max_depth)
    images.append(pred_depth_image)
    labels.append(shape_labels[shape_i] + f"\nScore={round_str(score,4)}" )

dst = multi_panel(images, labels, middle_width, top_border, 13)


dst.save(f"test_{test_idx}_best_img2.png") 



from IPython import embed; embed()
