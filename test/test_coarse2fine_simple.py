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
from jax3dp3.viz import save_depth_image, get_depth_image
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
    rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
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

Test Shapes: (ONLY ONE GT image)
Rectangular prism 0: (cube)
Rectangular prism 1: (near-cube)
Rectangular prism 2: (2 similar dims, 1 unsimilar dim)
Rectangular prism 3: (3 unsimilar dims)
Rectangular prism 4: (near-(square)plane)

Test GT Poses:
A single default pose
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
shapes_dims = jnp.array([jnp.array([0.5, 0.5, 0.5]), jnp.array([0.45, 0.5, 0.55]), jnp.array([0.40, 0.42, 0.8]), jnp.array([0.3, 0.5, 0.7]), jnp.array([0.05, 0.5, 0.5])])
shapes = [get_rectangular_prism_shape(shape_dims) for shape_dims in shapes_dims]
num_shapes = len(shapes_dims)
GT_SHAPE_ID = 0
print(f"Testing {num_shapes} shapes; ONE GT SHAPE {GT_SHAPE_ID}")

# generate test poses and corresponding gt shapes (the ith pose has (i%num_shapes)th shape)
f_jit = jax.jit(jax.vmap(lambda t: jnp.vstack(
                [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]))
             )
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.4, 0.4, 0.5, 3, 3, 3))
translations = jnp.einsum("ij,ajk->aik", default_pose, pose_deltas)
rotations = get_rotation_proposals(8,8)
gt_test_poses = jnp.array([default_pose]) 
print(f"Testing {len(gt_test_poses)} images (GT Poses)")

# generate singular test image
def render_test_images(gt_test_poses):
    images = []
    for test_idx, gt_test_pose in enumerate(gt_test_poses):
        gt_shape_id = GT_SHAPE_ID
        gt_test_image = render_planes(gt_test_pose, shapes[gt_shape_id],h,w,fx_fy,cx_cy) 
        images.append(gt_test_image)
    
    return jnp.array(images)
gt_test_images = render_test_images(gt_test_poses)
print(f"Rendered {len(gt_test_images)} gt test images")


def get_c2f_func(shape_id):
    scorer_for_shape = partial(scorer, get_rectangular_prism_shape(shapes_dims[shape_id]))
    coarse_to_fine_func = partial(coarse_to_fine_pose_and_weights, scorer_for_shape, enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r)
    return coarse_to_fine_func

def get_prediction(test_idx,x):
    gt_test_pose, gt_test_image = gt_test_poses[test_idx], gt_test_images[test_idx]
    latent_pose_estimate = gt_test_pose   # TODO always center at gt?

    # retrieve best pose hypothesis for each possible shape
    def _get_best_score_for_shape(shape_i, x):
        best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape = get_c2f_func(shape_i)(latent_pose_estimate, gt_test_image)             

        return shape_i + 1, jnp.array([best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape])  # return pose idx, best score
    _, best_pose_and_scores = jax.lax.scan(_get_best_score_for_shape, 0, shapes)

    best_shape_idx = jnp.argmax(best_pose_and_scores[:, -1])  # get highest-score shape
    
    pred = jnp.array([best_shape_idx, *best_pose_and_scores[best_shape_idx]])
    
    # save_with_jit(pred)
    
    return test_idx+1, pred

_, pred = jax.lax.scan(get_prediction, 0, gt_test_poses)

# calculate actual pose from `pred`
pred_poses = jax.vmap(lambda tr_pred,rot_pred: default_pose @ enumeration_grid_tr[-1][tr_pred] @ enumeration_grid_r[-1][rot_pred], in_axes=(0,0),out_axes=0)(pred[:, 1].astype('int32'), pred[:, 2].astype('int32'))




from IPython import embed; embed()
