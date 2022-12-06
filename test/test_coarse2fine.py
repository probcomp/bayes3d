import sys
sys.path.append('.')

from functools import partial 
import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.coarse_to_fine import coarse_to_fine
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.enumerations import get_rotation_proposals, make_translation_grid_enumeration
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points
from jax3dp3.viz.img import save_depth_image, get_depth_image
from jax3dp3.likelihood import threedp3_likelihood
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

render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)
render_planes_parallel_jit = jax.jit(jax.vmap(render_planes_lambda))



## Scorer generations

def scorer(shape, pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight

NUM_BATCHES = 4
def get_batched_scorer_for_shape(shape):
    scorer_parallel = jax.vmap(partial(scorer, shape), in_axes = (0, None, None))
    batched_scorer_parallel = partial(batched_scorer_parallel, scorer_parallel, NUM_BATCHES)
    
    return batched_scorer_parallel



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
schedule_tr_fine = [(finest_r, coarsest_gridrange, coarsest_gridrange//finest_grid_resolution)]
schedule_rot_fine = [schedule_rot[-1]]

enumeration_likelihood_r_fine = [schedule_tr_fine[-1][0]]
enumeration_grid_tr_fine = [make_translation_grid_enumeration(
                        -grid_width, -grid_width, -grid_width,
                        grid_width, grid_width, grid_width,
                        num_grid_points,num_grid_points,num_grid_points
                    ) for (_, grid_width, num_grid_points) in schedule_tr_fine]
enumeration_grid_r_fine = [get_rotation_proposals(fib_nums, rot_nums) for (fib_nums, rot_nums) in schedule_rot_fine]

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



TEST_RESULT_DICT = {shape:{"frames_tested": 0, 
                            "true_positive": 0, 
                            "false_positive": 0, 
                            "false_negative":0} for shape in range(5)}
TOTAL_TIME_ELAPSED_C2F = 0
TOTAL_TIME_ELAPSED_FINE = 0

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

# generate test poses and corresponding gt shapes (the ith pose has (i%5)th shape)
f_jit = jax.jit(jax.vmap(lambda t: jnp.vstack(
                [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]))
             )
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.4, 0.4, 0.5, 3, 3, 3))
translations = jnp.einsum("ij,ajk->aik", default_pose, pose_deltas)
rotations = get_rotation_proposals(8,8)
gt_test_poses = jnp.einsum("aij,bjk->abik", translations, rotations).reshape(-1, 4,4)



## Generate c2f functions to jit and run test

coarse_to_fine_functions = []
for shape in shapes:
    batched_scorer_parallel = get_batched_scorer_for_shape(shape)
    coarse_to_fine = partial(coarse_to_fine, batched_scorer_parallel)
    coarse_to_fine_jit = jax.jit(coarse_to_fine)
    
    coarse_to_fine_functions.append(coarse_to_fine_jit)

for i, gt_test_pose in enumerate(gt_test_poses):
    best_poses = np.zeros((num_shapes, 4, 4))
    best_pose_scores = np.zeros(num_shapes)

    gt_shape = shapes[i % num_shapes]
    gt_test_image = render_planes_jit(gt_test_pose, gt_shape)
    latent_pose_estimate = gt_test_pose   # TODO always center at gt?
    # retrieve best pose hypothesis for each possible shape
    for shape_i, shape in enumerate(shapes):
        coarse_to_fine_jit = coarse_to_fine_functions[shape_i]

        _ = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r,
                                jnp.array([
                                [1.0, 0.0, 0.0, 0.0],   
                                [0.0, 1.0, 0.0, 0.0],   
                                [0.0, 0.0, 1.0, 0.0],   
                                [0.0, 0.0, 0.0, 0.0],   
                                ]), jnp.zeros(gt_test_image.shape))

        start = time.time()                       
        best_pose_estimate = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r, latent_pose_estimate, gt_test_image)             
        end = time.time()
        print(best_pose_estimate)
        elapsed = end - start
        TOTAL_TIME_ELAPSED_C2F += elapsed
        print("time elapsed = ", elapsed, " FPS=", 1/elapsed)
        best_img = render_planes_jit(best_pose_estimate)
        # save_depth_image(best_img[:,:,2], f"c2f_out.png",max=max_depth)

        best_poses[shape_i, :, :] = best_pose_estimate
        best_pose_score = scorer(shape, best_pose_estimate, gt_test_image, enumeration_likelihood_r[-1])  # best pose's likelihood at finest resolution
        best_pose_scores[shape_i] = best_pose_score
    
        # TODO run fine-scale enumeration for runtime comparison
        start = time.time()                       
        _ = coarse_to_fine_jit(enumeration_grid_tr_fine, enumeration_grid_r_fine, enumeration_likelihood_r, latent_pose_estimate, gt_test_image)             
        end = time.time()
        elapsed = end - start
        TOTAL_TIME_ELAPSED_FINE += elapsed

    
    # Test accuracy: get best shape-pose pair for image
    best_shape_for_gt_pose = jnp.argmax(best_pose_scores)
    TEST_RESULT_DICT[i % num_shapes]["frames_tested"] += 1
    if best_shape_for_gt_pose == (i % num_shapes):
        TEST_RESULT_DICT[i % num_shapes]["true_positive"] += 1
    else:
        TEST_RESULT_DICT[i % num_shapes]["false_negative"] += 1
        TEST_RESULT_DICT[best_shape_for_gt_pose]["false_positive"] += 1

# Test runtime: 
TIME_ELAPSED_PER_FRAME_C2F = TOTAL_TIME_ELAPSED_C2F / (num_shapes * len(gt_test_poses))
# TODO fine-enum-only time recording

from IPython import embed; embed()
