import collections
import heapq
import jax
import jax.numpy as jnp
import bayes3d
import bayes3d as b
import bayes3d.transforms_3d as t3d
import numpy as np
import heapq

####################################################################################
# Scheduling functions
####################################################################################

def make_schedules_contact_params(grid_widths, rotation_angle_widths, grid_params):
    ## version of make_schedules with angle range reduction based on previous iter
    sched = []

    for (grid_width, angle_width, grid_param) in zip(grid_widths, rotation_angle_widths, grid_params):
        cf = bayes3d.scene_graph.enumerate_contact_and_face_parameters(
            -grid_width, -grid_width, -angle_width, 
            +grid_width, +grid_width, angle_width, 
            *grid_param,  # *grid_param is num_x, num_y, num_angle
            jnp.arange(6)
        )
        sched.append(cf)
    return sched

def make_schedules_full_pose_params(grid_widths, sphere_angle_widths, rotation_angle_widths, grid_params):
    ## version of make_schedules with angle range reduction based on previous iter
    sched = [] 
    for (grid_width, sphere_angle_width, rotation_angle_width, grid_param) in zip(grid_widths, sphere_angle_widths, rotation_angle_widths, grid_params):
        min_x, min_y, min_z = -grid_width, -grid_width, -grid_width
        max_x,max_y,max_z = grid_width, grid_width, grid_width
        num_x, num_y, num_z, num_fib_sphere, num_planar_angle = grid_param

        cf = bayes3d.enumerations.make_grid_enumeration(
            min_x, min_y, min_z, -rotation_angle_width,
            max_x,max_y,max_z, rotation_angle_width, 
            num_x,num_y,num_z, num_fib_sphere, num_planar_angle, 
            sphere_angle_width)
        sched.append(cf)
    return sched

def make_schedules(grid_widths, angle_widths, grid_params, full_pose=False, sphere_angle_widths=None):
    # check schedule validity
    assert len(grid_widths) == len(angle_widths) == len(grid_params)  
    
    if not full_pose: 
        assert len(grid_params[0]) == 3, "pass in (num_x, num_y, num_angles) as grid_param"
        return make_schedules_contact_params(grid_widths, angle_widths, grid_params)   
    else:
        assert len(grid_params[0]) == 5, "pass in (num_x, num_y, num_z, num_fib_sphere, num_planar_angle) as grid_param"
        if sphere_angle_widths is None: 
            sphere_angle_widths = [jnp.pi for _ in angle_widths]
        assert len(grid_widths) == len(sphere_angle_widths)
        return make_schedules_full_pose_params(grid_widths, sphere_angle_widths, angle_widths, grid_params)


contact_poses_parallel = jax.vmap(
    b.scene_graph.relative_pose_from_edge,
    in_axes=(0, None, None),
)
def c2f_iter_trace_contact_params(trace, init_contact_param, contact_param_deltas, contact_plane, box_dim, obj_id, face, VARIANCE_GRID, OUTLIER_GRID):
    contact_param_grid = contact_param_deltas + init_contact_param
    potential_new_object_poses = contact_plane @ contact_poses_parallel(
        contact_param_grid,
        face,
        box_dim,
    )
    potential_poses = jnp.concatenate(
        [
            jnp.tile(trace.poses[:,None,...], (1,potential_new_object_poses.shape[0],1,1)),
            potential_new_object_poses[None,...]
        ]
    )
    traces = b.Traces(
        potential_poses, jnp.concatenate([trace.ids, jnp.array([obj_id])]), VARIANCE_GRID, OUTLIER_GRID,
        trace.outlier_volume, trace.observation
    )
    p = b.score_traces(traces)

    ii,jj,kk = jnp.unravel_index(p.argmax(), p.shape)
    contact_param = contact_param_grid[ii]
    return contact_param, traces, traces[ii,jj,kk]

c2f_iter_trace_contact_params_jit = jax.jit(c2f_iter_trace_contact_params)