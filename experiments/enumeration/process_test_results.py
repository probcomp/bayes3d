import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax3dp3.enumerations import get_rotation_proposals, make_translation_grid_enumeration
from jax3dp3.metrics import get_rot_error_from_poses, get_translation_error_from_poses

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


x0, y0, z0 = 0.0, 0.0, 1.50
default_pose = jnp.array([
    [1.0, 0.0, 0.0, x0],   
    [0.0, 1.0, 0.0, y0],   
    [0.0, 0.0, 1.0, z0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

# load test results
with open('gt_test_poses.npy', 'rb') as f:
    gt_poses = jnp.load(open('gt_test_poses.npy', 'rb') )   # best idx, best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape
with open('predictions.npy', 'rb') as f1:
    predictions = jnp.load(open('predictions.npy', 'rb') )   # best idx, best_tr_idx_for_shape, best_rot_idx_for_shape, best_pose_score_for_shape

num_tests = len(gt_poses)
assert num_tests == len(predictions)


# recreate enumeration grids from test
# tuples of (radius, width of gridding, num gridpoints)
likelihood_r, grid_width, num_grid_points = (0.02, 0.1, 10)
fib_nums, rot_nums = (30, 30)

enumeration_grid_tr = make_translation_grid_enumeration(
                        -grid_width, -grid_width, -grid_width,
                        grid_width, grid_width, grid_width,
                        num_grid_points,num_grid_points,num_grid_points)
enumeration_grid_rot = get_rotation_proposals(fib_nums, rot_nums)


# process predictions ([best_shape_idx, best_tr_idx, best_rot_idx, best_score] in enum) into actual poses
pred_ids = predictions[:,0]
pred_poses = jax.vmap(lambda tr_pred,rot_pred: default_pose @ enumeration_grid_tr[tr_pred] @ enumeration_grid_rot[rot_pred], in_axes=(0,0),out_axes=0)(predictions[:, 1].astype('int32'), predictions[:, 2].astype('int32'))

def get_classification_accuracy(pred_shapes, gt_shapes):
    return pred_shapes == gt_shapes


def get_translation_errors(pred_poses, gt_poses):
    raise NotImplementedError("See jax3dp3/metrics: get_translation_error_from_poses()") 

def get_rotation_errors(pred_poses, gt_poses):
    vmap_acc = jax.vmap(get_rot_error_from_poses, in_axes=(0,0), out_axes=0)  # TODO adjust symmetry
    return vmap_acc(pred_poses, gt_poses)


## 
# get test results
##
NUM_TEST_SHAPES = 5
gt_indices = jnp.arange(num_tests) % NUM_TEST_SHAPES  # alternated between shapes for gt image generation

classification_accurate_num = get_classification_accuracy(pred_ids, gt_indices).sum()
print(f"accuracy = {classification_accurate_num}/{num_tests}")

from IPython import embed; embed()

rot_errors = get_rotation_errors(pred_poses, gt_poses)
print(f"mean rotation error (radians)={rot_errors.mean()}")
# TODO account for symmetry (e.g. pi radians error = 0 error along an axis)


from IPython import embed; embed()
