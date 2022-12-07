import sys
sys.path.append('.')

from functools import partial 
import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.coarse_to_fine import coarse_to_fine
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

# fine-only "schedule": enumerate over same range as initial coarse sweep of c2f, 
# but with the finest likelihood/grid resolution of c2f (i.e. more gridpoints)
finest_r = schedule_tr[-1][0]  # 0.02
coarsest_gridrange = schedule_tr[0][1]  # 1
finest_grid_resolution = schedule_tr[-1][1] / schedule_tr[-1][-1]  # 0.1/10=0.01
schedule_tr_fine = [(finest_r, coarsest_gridrange, int(coarsest_gridrange/finest_grid_resolution))]
schedule_rot_fine = [schedule_rot[-1]]

# enumeration_likelihood_r_fine = [schedule_tr_fine[-1][0]]
# enumeration_grid_tr_fine = [make_translation_grid_enumeration(
#                         -grid_width, -grid_width, -grid_width,
#                         grid_width, grid_width, grid_width,
#                         num_grid_points,num_grid_points,num_grid_points
#                     ) for (_, grid_width, num_grid_points) in schedule_tr_fine]
# enumeration_grid_r_fine = [get_rotation_proposals(fib_nums, rot_nums) for (fib_nums, rot_nums) in schedule_rot_fine]

######
# Setup testing (see test/test_generate_enumerations.py)

# Test Shapes:
# Rectangular prism 0: (cube)
# Rectangular prism 1: (near-cube)
# Rectangular prism 2: (2 similar dims, 1 unsimilar dim)
# Rectangular prism 3: (3 unsimilar dims)
# Rectangular prism 4: (near-(square)plane)

# Test GT Poses:
# 4x4x4 translation enumerations
# 8x8 rotation enumerations
# (4096 pose proposals to test)
######

# generate test shapes
x0, y0, z0 = 0.0, 0.0, 1.50
default_pose = jnp.array([
    [1.0, 0.0, 0.0, x0],   
    [0.0, 1.0, 0.0, y0],   
    [0.0, 0.0, 1.0, z0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

shapes_dims = [jnp.array([0.5, 0.5, 0.5]), jnp.array([0.45, 0.5, 0.55]), jnp.array([0.40, 0.42, 0.8]), jnp.array([0.3, 0.5, 0.7]), jnp.array([0.05, 0.5, 0.5])]
shapes = [get_rectangular_prism_shape(shape_dims) for shape_dims in shapes_dims]
num_shapes = len(shapes)
print(f"Testing {num_shapes} shapes")

# generate test poses and corresponding gt shapes (the ith pose has (i%5)th shape)
f_jit = jax.jit(jax.vmap(lambda t: jnp.vstack(
                [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]))
             )
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.4, 0.4, 0.5, 3, 3, 3))
translations = jnp.einsum("ij,ajk->aik", default_pose, pose_deltas)
rotations = get_rotation_proposals(8,8)
gt_test_poses = jnp.einsum("aij,bjk->abik", translations, rotations).reshape(-1, 4,4)
print(f"Testing {len(gt_test_poses)} images (GT Poses)")


def get_renderer(shape_id, h=h,w=w,fx_fy=fx_fy,cx_cy=cx_cy):
    render_planes_lambda = lambda p: render_planes(p,shapes[shape_id],h,w,fx_fy,cx_cy)
    return render_planes_lambda

def get_c2f_func(shape_id):
    scorer_for_shape = partial(scorer, shapes[shape_id])
    coarse_to_fine_func = partial(coarse_to_fine, scorer_for_shape, enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r)
    return coarse_to_fine_func

def get_predictions(shapes, gt_test_poses):   
    min_r = 0.02

    test_predictions = []

    for i, gt_test_pose in enumerate(gt_test_poses):
        pose_estimates = []
        scores = []

        gt_shape_id = i % num_shapes
        gt_test_image = get_renderer(gt_shape_id)(gt_test_pose)
        latent_pose_estimate = gt_test_pose   # TODO always center at gt?
        # retrieve best pose hypothesis for each possible shape
        for shape_i, shape in enumerate(shapes):
            best_pose_for_shape = get_c2f_func(shape_i)(latent_pose_estimate, gt_test_image)             
            best_pose_score_for_shape = scorer(shape, best_pose_for_shape, gt_test_image, min_r)  # best pose's likelihood at finest resolution
            
            pose_estimates.append(best_pose_for_shape)
            scores.append(best_pose_score_for_shape)

        best_shape_idx = jnp.argmax(jnp.array(scores))
        best_pose_estimate = jnp.array(pose_estimates)[best_shape_idx]
        test_predictions.append((best_shape_idx, best_pose_estimate))
            
    return test_predictions
 
get_predictions_jit = jax.jit(get_predictions)
_ = get_predictions_jit(shapes, jnp.zeros((5,4,4)))

test_predictions = get_predictions_jit(shapes, gt_test_poses)

from IPython import embed; embed()
