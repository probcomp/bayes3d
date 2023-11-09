import genjax
import bayes3d as b
import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint
from dataclasses import dataclass
from genjax.generative_functions.distributions import ExactDensity

MAX_UNFOLD_LENGTH = 100

############################### V1 ########################################

@genjax.gen
def dynamics_v1(prev):
    """
    simple dynamics model where the params are the same every timestep
    """
    (t, pose, velocity, dynamics_params) = prev
    velocity = b.gaussian_vmf_pose(velocity, *dynamics_params[0])  @ "velocity"
    # pose = b.gaussian_vmf_pose(pose @ velocity, *dynamics_params[1])  @ f"pose"
    pose = pose @ velocity
    return (t + 1, pose, velocity, dynamics_params)

dynamics_v1_unfold = genjax.UnfoldCombinator.new(dynamics_v1, MAX_UNFOLD_LENGTH)
image_likelihood_vmap = genjax.MapCombinator.new(genjax.gen(lambda *x: b.image_likelihood(*x) @ "depths"), in_axes=(0,None,None,None,None))

@genjax.gen
def model_v1(T_vec, N_total_vec, N_vec, all_box_dims, pose_bounds, outlier_volume,
             init_vel_params, dynamics_params, variance_params, outlier_prob_params):
    """
    Multi Object with Unfold
    TODO: Each object needs a new starting position
    """
    T = T_vec.shape[0]
    # sample init pose and velocity
    pose = b.uniform_pose(pose_bounds[0], pose_bounds[1]) @ "init_pose"
    velocity = b.gaussian_vmf_pose(jnp.eye(4), *init_vel_params) @ "init_velocity"

    all_poses = []
    for i in range(N_vec.shape[0]):
        # sample dynamics over T time steps
        dynamics = dynamics_v1_unfold(T,(0, pose, velocity, dynamics_params)) @ f"dynamics_{i+1}"
        # Slice off pose from the full unfold memory
        poses = jax.lax.slice_in_dim(dynamics[1], 0, T+1, axis = 0) 
        all_poses.append(poses)
    
    all_poses = jnp.stack(all_poses, axis = 1)

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    rendered_images = b.RENDERER.render_many(
        all_poses, indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"
    
    images = image_likelihood_vmap(rendered_images, variance, outlier_prob, outlier_volume, 1.0) @ "depths"
    return rendered_images, poses

model_v1_simulate_jit = jax.jit(model_v1.simulate)
model_v1_importance_jit = jax.jit(model_v1.importance)


@genjax.gen
def model_v2(T_vec, N_total_vec, N_vec, all_box_dims, pose_bounds, outlier_volume,
             init_vel_params, dynamics_params, variance_params, outlier_prob_params):
    """
    Multi Object without Unfold
    TODO: EACH OBJECT needs a new starting position
    """
    T = T_vec.shape[0]
    # sample init pose and velocity
    pose = b.uniform_pose(pose_bounds[0], pose_bounds[1]) @ "init_pose"
    velocity = b.gaussian_vmf_pose(jnp.eye(4), *init_vel_params) @ "init_velocity"

    all_poses = []
    for i in range(N_vec.shape[0]):
        poses = [pose]
        for j in range(T):
            velocity = b.gaussian_vmf_pose(velocity, *dynamics_params[0])  @ f"velocity_{j+1}"
            pose = pose @ velocity
            poses.append(pose)
        
        all_poses.append(jnp.stack(poses))
    
    all_poses = jnp.stack(all_poses, axis = 1)

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    rendered_images = b.RENDERER.render_many(
        all_poses, indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"
    
    images = image_likelihood_vmap(rendered_images, variance, outlier_prob, outlier_volume, 1.0) @ "depths"
    return rendered_images, poses

model_v2_simulate_jit = jax.jit(model_v2.simulate)
model_v2_importance_jit = jax.jit(model_v2.importance)


# DEPRECATE
@genjax.gen
def HMM_state(prev, dynamics_params):
    (pose, velocity) = prev
    velocity = b.gaussian_vmf_pose(velocity, *dynamics_params[0])  @ "velocity"
    return pose @ velocity, velocity

# DEPRECATE
@genjax.gen
def model_v3(T_vec, N_total_vec, N_vec, all_box_dims, pose_bounds, outlier_volume,
             init_vel_params, dynamics_params, variance_params, outlier_prob_params,
             keys, chms):
    """
    Hidden Markov Model with there being separate traces for each state. The dynamics are 
    decoupled from the chain of the dynamic scene
    """
    T = T_vec.shape[0]
    # sample init pose and velocity
    pose = b.uniform_pose(pose_bounds[0], pose_bounds[1]) @ "init_pose"
    velocity = b.gaussian_vmf_pose(jnp.eye(4), *init_vel_params) @ "init_velocity"

    all_poses = []
    all_traces = []
    for i in range(N_vec.shape[0]):
        poses = [pose]
        traces = []
        for j in range(T):
            _, tr = HMM_state.importance(keys[j], chms[j], ((pose,velocity),dynamics_params))
            pose, velocity = tr.get_retval()
            poses.append(pose)
            traces.append(tr)
        
        all_poses.append(jnp.stack(poses))
        all_traces.append(traces)
    
    all_poses = jnp.stack(all_poses, axis = 1)

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    rendered_images = b.RENDERER.render_many(
        all_poses, indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"
    
    images = image_likelihood_vmap(rendered_images, variance, outlier_prob, outlier_volume, 1.0) @ "depths"
    return rendered_images, poses, all_traces

model_v3_simulate_jit = jax.jit(model_v3.simulate)
model_v3_importance_jit = jax.jit(model_v3.importance)


@genjax.gen
def model_v4(pose, velocity, N_total_vec, N_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params):
    """
    Single Object Model HMM
    """
    velocity = b.gaussian_vmf_pose(velocity, *vel_params)  @ "velocity"
    pose = pose @ velocity

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    single_rendered_image = b.RENDERER.render(
        pose[None,...], indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth"

    return single_rendered_image, (pose, velocity)

model_v4_simulate_jit = jax.jit(model_v4.simulate)
model_v4_importance_jit = jax.jit(model_v4.importance)


@genjax.gen
def model_v5(prev_state, N_total_vec, N_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params):
    """
    Single Object Model HMM
    """
    (_, pose, velocity) = prev_state
    velocity = b.gaussian_vmf_pose(velocity, *vel_params)  @ "velocity"
    pose = pose @ velocity

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    single_rendered_image = b.RENDERER.render(
        pose[None,...], indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth"

    return (single_rendered_image, pose, velocity)


def score_images_0(rendered, observed):
    return -jnp.linalg.norm(observed - rendered, axis=-1).mean()

def score_images_1(rendered, observed):
    mask = observed[...,2] < intrinsics.far
    return (jnp.linalg.norm(observed - rendered, axis=-1)* (1.0 * mask)).sum() / mask.sum()

def score_images_2(rendered, observed):
    return -jnp.linalg.norm(observed - rendered, axis=-1).mean()

def score_images_3(rendered, observed):
    distances = jnp.linalg.norm(observed - rendered, axis=-1)
    probabilities_per_pixel = jax.scipy.stats.norm.logpdf(
        distances,
        loc=0.0, 
        scale=0.02
    )
    image_probability = probabilities_per_pixel.mean()
    return image_probability

def score_images_4(rendered, observed):
    distances = jnp.linalg.norm(observed - rendered, axis=-1)
    width = 0.1
    probabilities_per_pixel = (distances < width/2) / width
    return probabilities_per_pixel.mean()

@dataclass
class DebugLikelihood(ExactDensity):
    def sample(self, key, img):
        return img

    def logpdf(self, image, s):
        return score_images_4(
            image, s)
    
debug_likelihood = DebugLikelihood()
    
@genjax.gen
def model_v5b(prev_state, N_total_vec, N_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params):
    """
    Single Object Model HMM
    """
    (_, pose) = prev_state
    pose = b.gaussian_vmf_pose(pose, *vel_params)  @ "velocity"

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    single_rendered_image = b.RENDERER.render(
        pose[None,...], indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth"

    return (single_rendered_image, pose)

@genjax.gen
def model_v5c(prev_state, N_total_vec, N_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params, occ_pose):
    """
    WITH OCCLUDER IDX for simple physics scene
    """
    (_, pose) = prev_state
    pose = b.gaussian_vmf_pose(pose, *vel_params)  @ "velocity"

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    all_poses = jnp.stack([pose,occ_pose])

    single_rendered_image = b.RENDERER.render(
        all_poses, jnp.array([0,1]) 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth"

    return (single_rendered_image, pose)

@genjax.gen
def model_v5d(prev_state, N_total_vec, N_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params, occ_pose):
    """
    WITH OCCLUDER IDX for simple physics scene
    """
    (_, prev_pose, pose) = prev_state
    # find pose change
    vel_pos = pose[:3,3] - prev_pose[:3,3]
    vel_pose = b.t3d.transform_from_pos(vel_pos)
    next_pose = vel_pose @ pose # in global frame
    pose = b.gaussian_vmf_pose(next_pose, *vel_params)  @ "velocity"

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    all_poses = jnp.stack([next_pose,occ_pose])

    single_rendered_image = b.RENDERER.render(
        all_poses, jnp.array([0,1]) 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth"

    return (single_rendered_image, pose, next_pose)

@genjax.gen
def model_v6(T_vec, outlier_volume,
            vel_params, variance_params, outlier_prob_params):
    """
    No unfold HMM-full
    """
    T = T_vec.shape[0]

    variance = genjax.distributions.tfp_uniform(*variance_params) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(*outlier_prob_params) @ "outlier_prob"

    pose = jnp.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,2.0],
        [0,0,0,1]
    ])

    single_rendered_image = b.RENDERER.render(
        pose[None,...], jnp.array([0]) 
    )[...,:3]

    image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ "depth_0"
    x,y,z = 0,0,0

    images = [image]
    deltas = [jnp.array([x,y,z])]

    for i in range(T):

        x = genjax.normal(0, vel_params[0]) @ f"x_{i+1}"
        y = genjax.normal(0, vel_params[1]) @ f"y_{i+1}"
        z = genjax.normal(0, vel_params[2]) @ f"z_{i+1}"

        velocity = jnp.array([
            [1,0,0,x],
            [0,1,0,y],
            [0,0,1,z],
            [0,0,0,1]
        ])

        pose = pose @ velocity

        single_rendered_image = b.RENDERER.render(
            pose[None,...], jnp.array([0]) 
        )[...,:3]

        image = b.image_likelihood(single_rendered_image, variance, outlier_prob, outlier_volume, 1.0) @ f"depth_{i+1}"
        images.append(image)
        deltas.append(jnp.array([x,y,z]))

    return jnp.stack(images), jnp.stack(deltas)

# model_v5_unfold = genjax.UnfoldCombinator.new(model_v5, MAX_UNFOLD_LENGTH)