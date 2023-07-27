import jax
import jax.numpy as jnp
from jax.debug import print as jprint
def physics_prior(proposed_pose, physics_estimated_pose):
    proposed_pos = proposed_pose[:3,3]
    physics_estimated_pos = physics_estimated_pose[:3,3]
    return jax.scipy.stats.multivariate_normal.logpdf(
        proposed_pos, physics_estimated_pos, jnp.diag(jnp.array([0.02, 0.02, 0.02]))
    )


physics_prior_parallel_jit = jax.jit(jax.vmap(physics_prior, in_axes=(0, None)))
physics_prior_parallel = jax.vmap(physics_prior, in_axes=(0, None))

def physics_prior_v1(prev_pose, prev_prev_pose, bbox_dims, camera_pose, world2cam):
    """
    Score the physics of the simulation outside of a PPL, this will
    score physics estimates independent of what we see (3DP3 likelihood)

    Version 1:

    ASSUMPTIONS:
    A1 - Single object
    A2 - Change in position, no change in orientation 
    A3 - No Friction
    A4 - No Restitution/Damping (no bouncing)
    A5 - No Collision between objects (Except single object and Floor)
    A6 - No Mass information
    A7 - No Acceleration in X-Y-Z direction
    A8 - Above line implies no Gravitational acceleration
    A9 - Centre of bbox is centre of of object's local frame
    A10 - bbox_dims are aligned with the world axes

    IMPLEMENTATION:
    I1 - Simple X-Y-Z translation integration from T-1 & T-2 (0 accleration in X-Y-Z directions)
    I2 - Switch off Y-translation if bottom of object hits the floor (assume floor_y = 0) - collision between floor and single object
    I3 - Multivariate Gaussian Noise Model to get the score

    INPUTS:
    prev_pose: inferred pose at time = T-1
    prev_prev_pose: inferred pose at time = T-2
    bbox_dims: bounding box dimensions of single object
    camera_pose: cam2world affine transformation
    world2cam: world to camera coordinate transformation
    """

    # Assuming all poses are in camera frame

    # extract x-y-z positions
    prev_pos = prev_pose[:3,3]
    prev_prev_pos = prev_prev_pose[:3,3]

    # find X-Y-Z velocity change

    # I1 & I2 -> find simple difference in world frame + check if object 
    # is on the floor and force it to have no downward vector
    # conversions to world frame
    prev_prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_prev_pos, 1], axis = None)
    prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_pos, 1], axis = None)
    vel_pos_world = prev_pos_world - prev_prev_pos_world
    # find object's bottom in world frame
    object_bottom = prev_pos_world[2] - 0.5*bbox_dims[2]

    vel_pos_world = jax.lax.cond(jnp.less_equal(object_bottom, 0.01 * bbox_dims[2]),
        lambda x: x.at[2].set(0),
        lambda x: x,
        vel_pos_world)

    pred_pos_world = prev_pos_world + vel_pos_world

    pred_pos = world2cam[:3,:] @ jnp.concatenate([pred_pos_world, 1], axis = None)
    
    # I1 -> Integrate X-Y-Z forward to current time step
    # jprint("pred pos: {}", camera_pose[:3,:] @ jnp.concatenate([pred_pos, 1], axis = None))
    physics_estimated_pose = jnp.copy(prev_pose) # orientation is the same
    physics_estimated_pose = physics_estimated_pose.at[:3,3].set(pred_pos)

    return physics_estimated_pose
    
physics_prior_v1_jit = jax.jit(physics_prior_v1)


def physics_prior_v2(prev_poses, bbox_dims, camera_pose, world2cam, T, t_interval = 1.0/60.0):
    """
    Score the physics of the simulation outside of a PPL, this will
    score physics estimates independent of what we see (3DP3 likelihood)

    Version 2:

    ASSUMPTIONS:
    A1 - Single object
    A2 - Change in position, no change in orientation 
    A3 - No Friction
    A4 - No Restitution/Damping (no bouncing)
    A5 - No Collision between objects (Except single object and Floor)
    A6 - No Mass information
    A7 - No Acceleration in X-Y direction
    A8 - There is Gravitational acceleration
    A9 - Centre of bbox is centre of of object's local frame
    A10 - bbox_dims are aligned with the world axes

    IMPLEMENTATION:
    I1 - Simple X-Y-Z translation integration from T-1 & T-2 (0 accleration in X-Y-Z directions)
    I2 - Switch off Y-translation if bottom of object hits the floor (assume floor_y = 0) - collision between floor and single object
    I3 - Multivariate Gaussian Noise Model to get the score

    INPUTS:
    prev_poses: list of all poses estimated before time step T
    bbox_dims: bounding box dimensions of single object
    camera_pose: cam2world affine transformation
    world2cam: world to camera coordinate transformation
    """

    # Assuming all poses are in camera frame

    # extract x-y-z positions
    prev_pos = prev_poses[T,...]
    prev_prev_pos = prev_poses[T-1,...]

    # find X-Y-Z velocity change

    # I1 & I2 -> find simple difference in world frame + check if object 
    # is on the floor and force it to have no downward vector
    # conversions to world frame
    prev_prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_prev_pos, 1], axis = None)
    prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_pos, 1], axis = None)
    vel_pos_world = prev_pos_world - prev_prev_pos_world
    # find object's bottom in world frame
    object_bottom = prev_pos_world[2] - 0.5*bbox_dims[2]

    vel_pos_world = jax.lax.cond(jnp.less_equal(object_bottom, 0.01 * bbox_dims[2]),
        lambda x: x, # x.at[2].set(0),
        lambda x: x,
        vel_pos_world)

    pred_pos_world = prev_pos_world + vel_pos_world

    pred_pos = world2cam[:3,:] @ jnp.concatenate([pred_pos_world, 1], axis = None)
    
    # I1 -> Integrate X-Y-Z forward to current time step
    # jprint("pred pos: {}", camera_pose[:3,:] @ jnp.concatenate([pred_pos, 1], axis = None))
    physics_estimated_pose = jnp.copy(prev_pose) # orientation is the same
    physics_estimated_pose = physics_estimated_pose.at[:3,3].set(pred_pos)

    return physics_estimated_pose
    
physics_prior_v2_jit = jax.jit(physics_prior_v2)