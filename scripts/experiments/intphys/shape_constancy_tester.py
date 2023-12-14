import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import sys
import jax
import json
import csv
import time
import pickle
import genjax
import bayes3d as b
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from viz import *
from utils import *
from mcs.mcs_utils import *
from PIL import Image
from io import BytesIO
import bayes3d.transforms_3d as t3d
from jax.debug import print as jprint
from tqdm import tqdm
from dataclasses import dataclass
from genjax._src.core.pytree.utilities import *
from tensorflow_probability.substrates import jax as tfp
from genjax.generative_functions.distributions import ExactDensity
import jax.tree_util as jtu
from genjax._src.core.transforms.incremental import NoChange, UnknownChange, Diff
import zmq
import pickle5
import zlib
console = genjax.pretty()

# model time!

def get_height_bounds(i, world_pose):
    # Half dimensions to get the corner points relative to the center
    rotation_matrix = world_pose[:3,:3]
    center = world_pose[:3,3]
    dimensions = b.RENDERER.model_box_dims[i]
    half_dims = dimensions / 2

    # Local corner points of the box in its local coordinate system
    local_corners = jnp.array([
        [-half_dims[0], -half_dims[1], -half_dims[2]],  # Lower rear left corner
        [ half_dims[0], -half_dims[1], -half_dims[2]],  # Lower rear right corner
        [-half_dims[0],  half_dims[1], -half_dims[2]],  # Lower front left corner
        [ half_dims[0],  half_dims[1], -half_dims[2]],  # Lower front right corner
        [-half_dims[0], -half_dims[1],  half_dims[2]],  # Upper rear left corner
        [ half_dims[0], -half_dims[1],  half_dims[2]],  # Upper rear right corner
        [-half_dims[0],  half_dims[1],  half_dims[2]],  # Upper front left corner
        [ half_dims[0],  half_dims[1],  half_dims[2]]   # Upper front right corner
    ])

    # Apply rotation to each corner point
    global_corners = jnp.stack([center + rotation_matrix @ corner for corner in local_corners])

    # Find the bottom-most point
    bottom_most_point_z = jnp.min(global_corners[:,2])
    top_most_point_z = jnp.max(global_corners[:,2])
    # distance from centre of bbox to bottom of bbox
    center_to_bottom_dist = center[2] - bottom_most_point_z
    return bottom_most_point_z,top_most_point_z, center_to_bottom_dist

def get_translation_direction(all_poses, t_init,t_full, t):
    direction = all_poses[t-1][:3,3] - all_poses[t_full][:3,3]
    direction = cam_pose[:3,:3] @ direction
    direction_xy = direction.at[2].set(0)

    normalized_direction_xy = jax.lax.cond(jnp.equal(jnp.linalg.norm(direction_xy), 0),
                                         lambda: direction_xy,
                                         lambda: direction_xy/jnp.linalg.norm(direction_xy))
    return normalized_direction_xy


# This model has to be recompiled for different # objects for now this is okay
@genjax.gen
def physics_stepper(all_poses, t, t_init, t_full, i, friction, gravity):
    # TODO: SAMPLING FRICTION SCHEME --> can be of a hmm style

    #################################################################
    # First let us consider timestep t-1
    #################################################################
    # Step 2: find world pose
    pose_prev = all_poses[t-1]
    pose_prev_world = cam_pose @ pose_prev

    # Step 3: check if we are already on the floor
    bottom_z, top_z, center_to_bottom = get_height_bounds(i, pose_prev_world)
    # within 20% of the object's height in world frame
    already_on_floor = jnp.less_equal(bottom_z,0.2 * (top_z - bottom_z))
    
    # Step 1: Find world velocity
    vel_pose_camera = jnp.linalg.solve(all_poses[t-2], all_poses[t-1])
    pre_vel_xyz_world = cam_pose[:3,:3] @ vel_pose_camera[:3,3]
    mag_xy = jnp.linalg.norm(pre_vel_xyz_world[:2])
    
    mag_xy_friction = mag_xy - friction * mag_xy

    # mag_xy_friction = jax.lax.cond(
    #     jnp.less_equal(jnp.abs(mag_xy_friction),3e-3),
    #     lambda:0.0,
    #     lambda:mag_xy_friction)
    
    mag_xy, gravity = jax.lax.cond(already_on_floor,lambda:(mag_xy_friction,gravity),lambda:(mag_xy, gravity))

    dir_xy_world = get_translation_direction(all_poses, t_init, t_full, t)

    # Step 7: Determine mag and gravity

    vel_xyz_world = mag_xy * dir_xy_world
    # Step 6: apply z axis change
    vel_xyz_world = vel_xyz_world.at[2].set(pre_vel_xyz_world[2] - gravity * 1./20)

    # Step 5: find peturbed velocity (equal to original norm) with random rotation
    perturbed_rot_pose = GaussianVMFPoseUntraced()(jnp.eye(4), *(1e-20, 10000.0))  @ "perturb"

    vel_xyz_world_perturbed = perturbed_rot_pose[:3,:3] @ vel_xyz_world # without friction

    vel_xyz_camera = inverse_cam_pose[:3,:3] @ vel_xyz_world_perturbed

    # Step 8: Get velocity update in camera frame
    vel = pose_prev.at[:3,3].set(vel_xyz_camera)

    # Step 9: Identify next pose
    next_pose = pose_prev.at[:3,3].set(pose_prev[:3,3] + vel[:3,3]) # trans only, no rot

    # Step 10: Ensure new bottom of object is above floor --> ground collision
    next_pose_world = cam_pose @ next_pose
    bottom_z,_,center_to_bottom = get_height_bounds(i, next_pose_world)
    next_pose = jax.lax.cond(
        jnp.less_equal(bottom_z,0),
        lambda:inverse_cam_pose @ next_pose_world.at[2,3].set(center_to_bottom),
        lambda:next_pose
    )

    return next_pose

def threedp3_likelihood_arijit(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = (distances < variance/2) / variance
    average_probability = 1 * probabilities_per_pixel.mean()
    return average_probability

threedp3_likelihood_arijit_vmap = jax.vmap(threedp3_likelihood_arijit, in_axes=(None,0,None,None))
threedp3_likelihood_arijit_double_vmap = jax.vmap(threedp3_likelihood_arijit, in_axes=(0,0,None,None))

def outlier_gaussian(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = jax.scipy.stats.norm.pdf(
        distances,
        loc=0.0, 
        scale=variance
    )
    average_probability = 0.01 * probabilities_per_pixel.sum()
    return average_probability

outlier_gaussian_double_vmap = jax.vmap(outlier_gaussian, in_axes=(0,0,None,None))
outlier_gaussian_vmap = jax.vmap(outlier_gaussian, in_axes=(None,0,None,None))

def outlier_gaussian_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = jax.scipy.stats.norm.pdf(
        distances,
        loc=0.0, 
        scale=variance
    )
    return 0.01 * probabilities_per_pixel

outlier_gaussian_per_pixel_vmap = jax.vmap(outlier_gaussian_per_pixel, in_axes=(None,0,None,None))

@dataclass
class ImageLikelihoodArijit(ExactDensity):
    def sample(self, key, img, variance, outlier_prob):
        return img

    def logpdf(self, observed_image, latent_image, variance, outlier_prob):
        # return threedp3_likelihood_arijit(
        #     observed_image, latent_image, variance, outlier_prob,
        # )        
        return outlier_gaussian(
            observed_image, latent_image, variance, outlier_prob,
        )
    
@dataclass
class GaussianVMFPoseUntraced(ExactDensity):
    def sample(self, key, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf(key, pose_mean, var, concentration)

    def logpdf(self, pose, pose_mean, var, concentration, **kwargs):
        return 0

@genjax.gen
def mcs_model(prev_state, t_inits, t_fulls, init_poses, full_poses, pose_update_params, variance, outlier_prob):
    """
    Single Object Model HMM
    """

    (_, _, poses, all_poses, friction, t, gravity) = prev_state

    # jprint("t = {}, f = {}",t, friction)
    num_objects = poses.shape[0]
    
    # for each object
    for i in range(num_objects):        

        poses = poses.at[i].set(jax.lax.cond(
            jnp.equal(t_fulls[i],t), # full pose at the correct time step
            lambda:full_poses[i], 
            lambda:poses[i]))
        
        poses = poses.at[i].set(jax.lax.cond(
            jnp.equal(t_inits[i],t), # init pose at the correct time step
            lambda:init_poses[i], 
            lambda:poses[i]))

        physics_prob = jnp.asarray(jax.lax.cond(jnp.greater_equal(t,t_fulls[i]+2),lambda:1,lambda:0), dtype=int)
        physics_pose = physics_stepper(all_poses[:,i,...], t, t_inits[i], t_fulls[i], i, friction, gravity) @ f"physics_{i}"
        final_pose, update_params = jax.lax.cond(physics_prob, lambda:(physics_pose, pose_update_params), lambda:(poses[i], (jnp.array([1e20,1e20,1e20]), 0.)))
                
        updated_pose = b.gaussian_vmf_pose(final_pose, *update_params)  @ f"pose_{i}"
        poses = poses.at[i].set(updated_pose)
        
    all_poses = all_poses.at[t].set(poses)
    rendered_image_obj = b.RENDERER.render(
        poses, jnp.arange(num_objects))[...,:3]

    # NOTE: gt_images_bg is a global variable here as it consumes too much memory for the trace
    rendered_image = splice_image(rendered_image_obj, gt_images_bg[t])

    sampled_image = ImageLikelihoodArijit()(rendered_image, variance, outlier_prob) @ "depth"
    # sampled_image = b.old_image_likelihood(rendered_image, 0.1, 0.001,1000,None) @ "depth"

    return (rendered_image, rendered_image_obj, poses, all_poses, friction, t+1, gravity)

def pose_update_v5(key, trace_, pose_grid, enumerator):
    num_splits = (pose_grid.shape[0] // 400) + 1
    all_weights = jnp.array([])
    for split_pose_grid in jnp.array_split(pose_grid, num_splits):
        weights = enumerator.enumerate_choices_get_scores(trace_, key, split_pose_grid)
        all_weights = jnp.hstack([all_weights, weights])
    sampled_idx = all_weights.argmax() # jax.random.categorical(key, weights)
    # jprint("weights = {}",all_weights)
    # jprint("weight mix:{}",jnp.unique(jnp.sort(all_weights), size = 10))
    # jprint("idx chosen = {}",sampled_idx)
    return *enumerator.update_choices_with_weight(
        trace_, key,
        pose_grid[sampled_idx]
    ), pose_grid[sampled_idx]


pose_update_v5_jit = jax.jit(pose_update_v5, static_argnames=("enumerator",))


def c2f_pose_update_v5(key, trace_, reference, gridding_schedule, enumerator, obj_id,):
    for i in range(len(gridding_schedule)):
        updated_grid = jnp.einsum("ij,ajk->aik", reference, gridding_schedule[i])
        # Time to check valid poses that dont intersect with the floor
        valid = jnp.logical_not(are_bboxes_intersecting_many_jit(
                            (100,100,20),
                            b.RENDERER.model_box_dims[obj_id],
                            jnp.eye(4).at[:3,3].set([0,0,-10.1]),
                            jnp.einsum("ij,ajk->aik",cam_pose,updated_grid)
                            ))
        # if pose is not valid, use the reference pose
        valid_grid = jnp.where(valid[:,None,None], updated_grid, reference[None,...])
        weight, trace_, reference = pose_update_v5_jit(key, trace_, valid_grid, enumerator)
        # jprint("ref position is {}", reference[:3,3])

    return weight, trace_

# def c2f_pose_update_v5(key, trace_, reference, gridding_schedule, enumerator, obj_id,):

#     def c2f_body(carry, grid):
#         trace_, reference = carry
#         updated_grid = jnp.einsum("ij,ajk->aik", reference, grid)
#         # Time to check valid poses that dont intersect with the floor
#         valid = jnp.logical_not(are_bboxes_intersecting_many_jit(
#                             (100,100,20),
#                             b.RENDERER.model_box_dims[obj_id],
#                             jnp.eye(4).at[:3,3].set([0,0,-10.1]),
#                             jnp.einsum("ij,ajk->aik",cam_pose,updated_grid)
#                             ))
#         # if pose is not valid, use the reference pose
#         valid_grid = jnp.where(valid[:,None,None], updated_grid, reference[None,...])
#         weight, trace_, reference = pose_update_v5_jit(key, trace_, valid_grid, enumerator)
#         # jprint("ref position is {}", reference[:3,3])
#         return  (trace_, reference), weight

#     (trace_, _), weights = jax.lax.scan(c2f_body, (trace_, reference),gridding_schedule)

#     return weights[-1], trace_

c2f_pose_update_v5_vmap_jit = jax.jit(jax.vmap(c2f_pose_update_v5, in_axes=(0,0,None,None,None)),
                                    static_argnames=("enumerator", "obj_id"))

c2f_pose_update_v5_jit = jax.jit(c2f_pose_update_v5,static_argnames=("enumerator", "obj_id"))

def make_new_keys(key, N_keys):
    key, other_key = jax.random.split(key)
    new_keys = jax.random.split(other_key, N_keys)
    return key, new_keys

def update_choice_map_no_unfold(gt_depths, constant_choices, t):
    constant_choices['depth'] = gt_depths[t]
    return genjax.choice_map(
                constant_choices
            )

def argdiffs_modelv7(trace):
    """
    Argdiffs specific to mcs_single_obejct model with no unfold
    """
    args = trace.get_args()
    argdiffs = (
        jtu.tree_map(lambda v: Diff(v, UnknownChange), args[0]),
        *jtu.tree_map(lambda v: Diff(v, NoChange), args[1:]),
    )
    return argdiffs

def proposal_choice_map_no_unfold(addresses, args, chm_args):
    addr = addresses[0] # custom defined
    return genjax.choice_map({
                        addr: args[0]
            })

def resampling_priority_fn(particles, all_padded_idxs, t, outlier_variance=0.1):
    rendered = particles.get_retval()[0]
    padded_idxs = all_padded_idxs[t]
    max_rows, _ = padded_idxs.shape

    # Create a mask for valid indices (not padded)
    valid_mask = padded_idxs[:, 0] != -1  # Assuming -1 is the padding value

    # Use the mask to select valid indices, replace invalid indices with a default valid index (e.g., 0)
    valid_row_indices = jnp.where(valid_mask, padded_idxs[:, 0], 0)
    valid_col_indices = jnp.where(valid_mask, padded_idxs[:, 1], 0)

    # Gather rendered and ground truth values
    rendered_values_at_indices = rendered[:, valid_row_indices, valid_col_indices, 2]
    gt_values_at_indices = gt_images[t, valid_row_indices, valid_col_indices, 2]

    # Compute inliers, using the mask to ignore the contributions of invalid indices
    inliers = jnp.where(valid_mask, jnp.abs(rendered_values_at_indices - gt_values_at_indices[None, ...]) < outlier_variance, False)
    inliers_per_particle = jnp.sum(inliers, axis=1)

    log_probs = jnp.log(inliers_per_particle + 1e-9)  # Add a small constant to avoid log(0)

    eff_ss = ess(normalize_weights(log_probs))

    return eff_ss, log_probs

def inference_approach_G2(model, gt, gridding_schedules, model_args, init_state, key, t_start, constant_choices, friction_params, T, addr, n_particles):
    """
    Sequential Importance Sampling on the non-unfolded HMM model
    with 3D pose enumeration proposal

    WITH JUST ONE PARTICLE
    """
    
    num_objects = init_state[2].shape[0]

    def get_next_state(particle):
        return (None,None,*particle.get_retval()[2:])
    get_next_state_vmap = jax.vmap(get_next_state, in_axes = (0,))

    # sample friction
    key, friction_keys = make_new_keys(key, n_particles)
    # frictions = jax.vmap(genjax.normal.sample, in_axes = (0,None,None))(friction_keys,*friction_params)
    # frictions = jnp.linspace(-0.03,0.07,n_particles)
    qs = jnp.linspace(0.05,0.95,n_particles)
    frictions = tfp.distributions.Normal(*friction_params).quantile(qs)
    # frictions = frictions.at[n_particles-1].set(0.5)
    gravities = jnp.linspace(0.5,2,n_particles)
    # broadcast init_state to number of particles
    init_states = jax.vmap(lambda f,g:(*init_state[:4], f, init_state[4], g), in_axes=(0,0))(frictions, gravities)

    # define functions for SIS/SMC
    init_fn = jax.jit(jax.vmap(model.importance, in_axes=(0,None,0)))
    update_fn = jax.jit(model.update)
    proposal_fn = c2f_pose_update_v5_jit

    def smc_body(carry, t):
        # get new keys
        print("jit compiling")
        # initialize particle based on last time step
        jprint("t = {}",t)
        
        key, log_weights, states,  = carry
        key, importance_keys = make_new_keys(key, n_particles)
        key, resample_key = jax.random.split(key)
        key, proposal_key = jax.random.split(key)

        variance = jax.lax.cond(
            jnp.less_equal(t, t_start + 2),
            lambda: 2 * model_args[5],
            lambda: model_args[5]
        )

        modified_model_args = (*model_args[:5], variance, *model_args[6:])

        full_args = jax.vmap(lambda x,y:(x, *y), in_axes=(0,None))(states, modified_model_args)

        importance_log_weights, particles = init_fn(importance_keys, update_choice_map_no_unfold(gt,constant_choices, t), full_args)

        # propose good poses based on proposal
        def proposer(carry, p):
            key, idx = carry
            proposal_log_weight = 0
            # argdiff and enumerator
            argdiffs = argdiffs_modelv7(p)
            enumerators = [b.make_enumerator([(addr + f'_{i}')], 
                                        chm_builder = proposal_choice_map_no_unfold,
                                        argdiff_f=lambda x: argdiffs
                                        ) for i in range(num_objects)] 
            for obj_id in range(num_objects):
                key, new_key = jax.random.split(key)
                reference = jax.lax.cond(jnp.equal(t,model_args[1][obj_id]),
                                         lambda:model_args[3][obj_id],
                                         lambda:states[2][idx][obj_id])
                w, p = proposal_fn(new_key, p, reference, gridding_schedules[obj_id], enumerators[obj_id], obj_id)
                proposal_log_weight += w
            return (new_key, idx + 1), (proposal_log_weight, p)
        _, (proposal_log_weights, particles) = jax.lax.scan(proposer, (proposal_key, 0), particles)

        eff_ss, priority_fn_log_probs = resampling_priority_fn(particles, padded_all_obj_indices, t)

        # jprint("t = {}, ess = {}", t, eff_ss)

        # # Resampling when ess is below threshold
        indices = jax.lax.cond(eff_ss <= 0.9*n_particles,
                               lambda: jax.random.categorical(resample_key, priority_fn_log_probs, shape=(n_particles,)),
                               lambda: jnp.arange(n_particles))
        particles = jtu.tree_map(lambda v: v[indices], particles)

        # get weights of particles
        new_log_weight = log_weights + importance_log_weights
        next_states = get_next_state_vmap(particles)

        return (key, new_log_weight, next_states), (particles, indices)

    (_, final_log_weight, _), (particles, indices) = jax.lax.scan(
        smc_body, (key, jnp.zeros(n_particles), init_states), jnp.arange(t_start, T))
    rendered = particles.get_retval()[0]
    rendered_obj = particles.get_retval()[1]
    inferred_poses = particles.get_retval()[2]
    print("SCAN finished")
    return final_log_weight, rendered, rendered_obj, inferred_poses, particles, indices

def determine_plausibility_shape(results, offset = 3, rend_fraction_thresh = 0.75):
    # check to see if object is falling from top

    T = results['resampled_indices'].shape[0] - offset
    tsteps_before_start = results['inferred_poses'].shape[0] - T

    height, width = results['intrinsics'].height, results['intrinsics'].width
    starting_indices = results['all_obj_indices'][tsteps_before_start - offset]
    if starting_indices is not []:
        mean_i, mean_j = np.median(starting_indices[:,0]), np.median(starting_indices[:,1])
        from_top = (mean_i < height/2) and (mean_j > mean_i) and (mean_j < width -mean_i)
    else:
        from_top = False

    # first get base indices to reflect resampled particles
    n_particles = results['resampled_indices'].shape[1]
    resample_bools = np.all(results['resampled_indices'] == np.arange(n_particles), axis = 1)
    base_indices = np.arange(n_particles)
    for i in range(results['resampled_indices'].shape[0]):
        base_indices = base_indices[results['resampled_indices'][i]]
    # then get the rendering scores based on resampled_indices
    rend = np.array(results["rend_ll"][offset:,base_indices])
    # get the worst rendered scores (object-less)
    WR = results["worst_rend"][offset:]
    # flatten rend to get the best vector across time
    rend = np.max(rend, axis = 1)

    max_rend_possible = height * width * jax.scipy.stats.norm.pdf(
        0.,
        loc=0.0, 
        scale=results["variance"]
    ) * 0.01

    t_violation = None
    plausibility_list = [True for _ in range(tsteps_before_start)]
    plausible = True
    count_violation = 0
    for t in range(T):
        if WR[t] > rend[t]:
            plausible = False
            if t_violation is None:
                t_violation = tsteps_before_start + t
        if WR[t] < max_rend_possible and WR[t] == rend[t]:
            print(t, WR[t], rend[t])
            count_violation +=1
            if t_violation is None:
                t_violation = tsteps_before_start + t
        # if from_top and rend[t] > WR[t] and WR[t] >= WR[t-1] and t > T/2 and results['inferred_poses'].shape[0] < 220: # CONSIDER REMOVING THIS HACK
        #     WR_gap = max_rend_possible - WR[t]
        #     rend_gap = max_rend_possible - rend[t]
        #     rend_likelihood_fraction = (WR_gap - rend_gap)/WR_gap
        #     if rend_likelihood_fraction < rend_fraction_thresh:
        #         plausible = False
        #         if t_violation is None:
        #             t_violation = tsteps_before_start + t

        plausibility_list.append(plausible)

    if count_violation > 3:
        plausible = False
        plausibility_list = [t < t_violation for t in range(T)]
        
    return plausible, t_violation, plausibility_list, from_top

scene_ID = sys.argv[1]

import subprocess
import zlib
import sys
import machine_common_sense as mcs
import glob


class MCS_Observation:
    def __init__(self, rgb, depth, intrinsics, segmentation, cam_pose):
        """RGBD Image
        
        Args:
            rgb (np.array): RGB image
            depth (np.array): Depth image
            camera_pose (np.array): Camera pose. 4x4 matrix
            intrinsics (b.camera.Intrinsics): Camera intrinsics
            segmentation (np.array): Segmentation image
        """
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.segmentation  = segmentation
        self.cam_pose = cam_pose

def get_obs_from_step_metadata(step_metadata, intrinsics, cam_pose):
    rgb = np.array(list(step_metadata.image_list)[-1])
    depth = np.array(list(step_metadata.depth_map_list)[-1])
    seg = np.array(list(step_metadata.object_mask_list)[-1])
    colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
    seg_final = seg_final_flat.reshape(seg.shape[:2])
    observation = MCS_Observation(rgb, depth, intrinsics, seg_final, cam_pose)
    return observation

def cam_pose_from_step_metadata(step_metadata):
    cam_pose_diff_orientation = np.array([
        [ 1,0,0,0],
        [0,0,-1,-4.5], # 4.5 is an arbitrary value
        [ 0,1,0,step_metadata.camera_height],
        [ 0,0,0,1]
    ])
    inv_cam_pose = np.linalg.inv(cam_pose_diff_orientation)
    inv_cam_pose[1:3] *= -1
    cam_pose = np.linalg.inv(inv_cam_pose)
    return cam_pose

def intrinsics_from_step_metadata(step_metadata):
    width, height = step_metadata.camera_aspect_ratio
    aspect_ratio = width / height
    cx, cy = width / 2.0, height / 2.0
    fov_y = np.deg2rad(step_metadata.camera_field_of_view)
    fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
    fx = cx / np.tan(fov_x / 2.0)
    fy = cy / np.tan(fov_y / 2.0)
    clipping_near, clipping_far = step_metadata.camera_clipping_planes
    intrinsics = {
        'width' : width,
        'height' : height,
        'cx' : cx,
        'cy' : cy,
        'fx' : fx,
        'fy' : fy,
        'near' : clipping_near,
        'far' : clipping_far
    }
    return intrinsics

# file = "/home/ubuntu/arijit/bayes3d/scripts/experiments/intphys/mcs/eval_7/" + f"{scene_ID}.json"
# controller = mcs.create_controller("/home/ubuntu/arijit/bayes3d/scripts/experiments/intphys/mcs/config_level2.ini")
# scene_data = mcs.load_scene_json_file(file)
# step_metadata = controller.start_scene(scene_data)
# scene_intrinsics = intrinsics_from_step_metadata(step_metadata)
# scene_cam_pose = cam_pose_from_step_metadata(step_metadata)
# MCS_Observations = [get_obs_from_step_metadata(step_metadata, scene_intrinsics, scene_cam_pose)]

# def MCS_stepper():
#     while True:
#         yield

# for _ in tqdm(MCS_stepper()):
#     step_metadata = controller.step("Pass")
#     if len(step_metadata.action_list) == 0:
#         break
#     MCS_Observations.append(get_obs_from_step_metadata(step_metadata, scene_intrinsics, scene_cam_pose))  # Do stuff here

# print("Observations loaded")
# # scene_ID = np.random.randint(79384572398)
# print(f"Scene ID randomly generated as {scene_ID}")
# np.savez(f"/home/ubuntu/arijit/bayes3d/scripts/experiments/intphys/mcs/shape_npzs/{scene_ID}.npz", MCS_Observations)



print(f"Running {scene_ID}")
SCALE = 0.2
# observations = load_observations_npz(scene_ID)

## NOTE: NOTE: NOTE: CHANGE LINE BELOW FOR FINAL EVAL
observations = np.load('/home/ubuntu/arijit/bayes3d/scripts/experiments/intphys/mcs/shape_npzs' + "/{}.npz".format(scene_ID),allow_pickle=True)["arr_0"]
print(len(observations))
# observations = np.load("{}.npz".format(scene_ID),allow_pickle=True)["arr_0"]

preprocessed_data = preprocess_mcs_physics_scene(observations, MIN_DIST_THRESH=0.6, scale=SCALE)

(gt_images, gt_images_bg, gt_images_obj, intrinsics),\
(gt_images_orig, gt_images_bg_orig, gt_images_obj_orig, intrinsics_orig),\
registered_objects, obj_pixels, is_gravity, poses, cam_pose = preprocessed_data

inverse_cam_pose = jnp.linalg.inv(cam_pose)

# get obj indices padded
all_obj_indices = [np.argwhere(gt_images_obj[i,...,2] != intrinsics.far) for i in range(gt_images.shape[0])]
max_rows = max(obj_indices.shape[0] for obj_indices in all_obj_indices)
def pad_array(array, max_rows):
    padding = ((0, max_rows - array.shape[0]), (0, 0))  # Pad rows, not columns
    return jnp.pad(array, padding, constant_values=-1)

padded_all_obj_indices = jnp.stack([pad_array(array, max_rows) for array in all_obj_indices])

b.setup_renderer(intrinsics, num_layers= 1024)
for i,registered_obj in enumerate(registered_objects):
    b.RENDERER.add_mesh(registered_obj['mesh'])
    # f_p = registered_objects[i]["full_pose"]
    # registered_objects[i]["full_pose"] = f_p.at[2,3].set(f_p[2,3] + 0.5*b.RENDERER.model_box_dims[i][2])
if len(registered_objects) == 0:
    t_start = gt_images.shape[0]-100
    registered_objects.append({'t_init' : gt_images.shape[0]-100,
                            'pose' : jnp.eye(4).at[:3,3].set([0,0,1e+5]),
                            'full_pose' : jnp.eye(4).at[:3,3].set([0,0,1e+5]),
                            't_full' : gt_images.shape[0]-100})
    b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"), scaling_factor = 0.1)
else:
    t_start = np.min([x["t_full"] for x in registered_objects])
# video_from_rendered(gt_images, scale = int(1/SCALE), framerate=30)

height, width = intrinsics.height, intrinsics.width
starting_indices = all_obj_indices[t_start]
if starting_indices is not []:
    mean_i, mean_j = np.median(starting_indices[:,0]), np.median(starting_indices[:,1])
    from_top = (mean_i < height/2) and (mean_j > mean_i) and (mean_j < width -mean_i)
else:
    from_top = False

gridding_schedules = []
for box_dims in b.RENDERER.model_box_dims:
    c2fm1 = 2
    c2f0 = 1
    c2f1 = 0.35 * c2f0
    # c2f1 = 0.7 * c2f0
    c2f2 = 0.7 * c2f1
    c2f3 = 0.2 * c2f2
    c2f4 = 0.2 * c2f3
    c2f5 = 0.2 * c2f4
    c2f6 = 0.2 * c2f5

    c2fs = [c2f0,c2f1,c2f2,c2f3,c2f4]

    x,y,z = box_dims
    grid_widths = [[c2f*x, c2f*y, c2f*z] for c2f in c2fs]

    grid_nums = [(13,13,13),(7,7,7),(7,7,7),(7,7,7), (7,7,7)]
    gridding_schedule_trans = make_schedule_translation_3d_variable_grid(grid_widths, grid_nums)
    gridding_schedules.append(gridding_schedule_trans)

# Setup for inference
T = gt_images.shape[0]
num_registered_objects = len(registered_objects)
variance = 0.1
INIT_STATE = (
        None,
        None,
        jnp.tile(jnp.eye(4).at[2,3].set(1e+5)[None,...],(num_registered_objects,1,1)),
        jnp.zeros((T,num_registered_objects,4,4)),
        t_start
)
MODEL_ARGS = (
     jnp.array([r['t_init'] for r in registered_objects]),
     jnp.array([r['t_full'] for r in registered_objects]),
     jnp.array([r['pose'] for r in registered_objects]),
     jnp.array([r['full_pose'] for r in registered_objects]),
     (jnp.array([1e-0,1e-0,5e-1]), 5e-1),
     variance,
     None
)
CONSTANT_CHOICES = {}

if from_top:
    friction_params = (0.7,0.1)
else:
    friction_params = (0.02,0.05)

key = jax.random.PRNGKey(np.random.randint(0,2332423432))
if from_top:
    n_particles = 3
else:
    n_particles = 30
model = mcs_model

if is_gravity:
    print(f"{scene_ID} is a gravity scene")
    plausible, plausibility_list, _ = gravity_scene_plausible(poses, intrinsics_orig, cam_pose, observations)
    print(f"{scene_ID} --> {plausible}")
else:
    start = time.time()
    lw, rendered, rendered_obj, inferred_poses, trace, indices = inference_approach_G2(model, gt_images, 
    gridding_schedules, MODEL_ARGS, INIT_STATE, key, t_start, CONSTANT_CHOICES,friction_params, T, "pose", n_particles)
    print ("FPS:", rendered.shape[0] / (time.time() - start))

    print("finished run")

    worst_rend = outlier_gaussian_double_vmap(gt_images[t_start:], gt_images_bg[t_start:], variance,None)

    dummy_poses = np.tile(jnp.eye(4).at[2,3].set(-1e5)[None,None,None,...], (t_start,n_particles,num_registered_objects,1,1))
    concat_inferred_poses = np.concatenate([dummy_poses, inferred_poses])

    rend_ll = trace.project(genjax.select(("depth")))
    phy_ll = [trace.project(genjax.select((f"pose_{i}"))) for i in range(num_registered_objects)]

    data = {"rend_ll":rend_ll, "phy_ll":phy_ll, "all_obj_indices" :all_obj_indices,
            "inferred_poses" : concat_inferred_poses,
            "resampled_indices" : indices, "heuristic_poses" : poses, "worst_rend":worst_rend,
            "intrinsics" : intrinsics, "variance" : variance}

    plausible, t_violation, plausibility_list, _ = determine_plausibility_shape(data)


shape_output = {}
shape_output['plausibility'] = plausible

best_p_idx = np.argmax(rend_ll[-1])
vars = []
fracs = []
for var in np.linspace(0.002,0.501,500):
    xx1 = threedp3_likelihood_arijit(gt_images[-1],gt_images[-1],var,None)
    xx2 = threedp3_likelihood_arijit(gt_images[-1],rendered[-1][best_p_idx],var,None)
    xx3 = threedp3_likelihood_arijit(gt_images[-1],gt_images_bg[-1],var,None)
    frac = (xx2-xx3)/(xx1-xx3)
    vars.append(var)
    fracs.append(frac)
shape_output['vars'] = vars
shape_output['fracs'] = fracs
shape_output["num_objects"] = num_registered_objects
shape_output['gt_image'] = gt_images[-1]
shape_output['gt_image_bg'] = gt_images_bg[-1]
shape_output['rendered'] = rendered[-1]

with open(file, 'r') as json_file:
    data = json.load(json_file)

shape_output['answer'] = data['goal']['answer']['choice']

print("scene validity is ",plausible, " and the fraction is ", fracs[98], f"for {num_registered_objects} objects")

with open(f'/home/ubuntu/arijit/bayes3d/scripts/experiments/intphys/mcs/shape_results/shape_result_{scene_ID}.pkl', 'wb') as file:
    pickle.dump(shape_output, file)