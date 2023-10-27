import genjax
import bayes3d as b
import jax
import jax.numpy as jnp
import numpy as np
from jax.debug import print as jprint

MAX_UNFOLD_LENGTH = 100

############################### V1 ########################################

@genjax.gen
def dynamics_v1(prev):
    (t, pose, velocity) = prev
    velocity = b.gaussian_vmf_pose(velocity, 0.005, 10000.0)  @ f"velocity"
    pose = b.gaussian_vmf_pose(pose @ velocity, 0.005, 10000.0)  @ f"pose"
    return (t + 1, pose, velocity)

dynamics_v1_unfold = genjax.UnfoldCombinator.new(dynamics_v1, MAX_UNFOLD_LENGTH)
image_likelihood_vmap = genjax.MapCombinator.new(genjax.gen(lambda *x: b.image_likelihood(*x) @ "depths"), in_axes=(0,None,None,None,None))

@genjax.gen
def model_v1(T_vec, N_total_vec, N_vec, all_box_dims, pose_bounds, outlier_volume, focal_length):
    """
    Single Object
    """
    T = T_vec.shape[0]
    # sample init pose and velocity
    pose = b.uniform_pose(jnp.array(pose_bounds[0]), jnp.array(pose_bounds[1])) @ "init_pose"
    velocity = b.gaussian_vmf_pose(jnp.eye(4), 0.01, 10000.0) @ "init_velocity"

    all_poses = []
    for i in range(N_vec.shape[0]):
        # sample dynamics over T time steps
        dynamics = dynamics_v1_unfold(T,(0, pose, velocity)) @ f"dynamics_{i+1}"
        # Slice off pose from the full unfold memory
        poses = jax.lax.slice_in_dim(dynamics[1], 0, T+1, axis = 0) 
        all_poses.append(poses)
    
    all_poses = jnp.stack(all_poses, axis = 1)

    indices = b.uniform_discrete_array(N_total_vec, N_vec) @ "indices"

    rendered_images = b.RENDERER.render_many(
        all_poses, indices 
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(0.00000000001, 10000.0) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(-0.01, 10000.0) @ "outlier_prob"
    
    images = image_likelihood_vmap(rendered_images, variance, outlier_prob, outlier_volume, focal_length) @ "depths"
    return rendered_images, poses

model_v1_simulate_jit = jax.jit(model_v1.simulate)
model_v1_importance_jit = jax.jit(model_v1.importance)