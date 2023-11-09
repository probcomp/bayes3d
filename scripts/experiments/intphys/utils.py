import bayes3d as b
import numpy as np
import genjax
import jax.numpy as jnp
import jax
import pickle
import os

def save_metadata(metadata, name, force_save = False):
    if not force_save and os.path.exists(f'{name}.pkl'):
        check = input(f"{name}.pkl already exists, do you want to overwrite? (y/n)")
        if 'n' in check.lower():
            print(f"{name}.pkl already exists")
            return
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(metadata, file)

def load_metadata(name):
    if '.pkl' != name[-4:]:
        name += '.pkl' 
    with open(name, 'rb') as file:
        return pickle.load(file)

def make_schedule_translation_3d(grid_widths, grid_nums):
    sched = []

    for (grid_width, grid_num) in zip(grid_widths, grid_nums):
        grid = b.utils.make_translation_grid_enumeration(
            -grid_width, -grid_width, -grid_width, 
            +grid_width, +grid_width, +grid_width, 
            *grid_num,  # *grid_num is num_x, num_y, num_z
        )
        sched.append(grid)
    return sched

def make_schedule_3d(grid_widths, grid_nums, rot_angle_bounds, fib_points, angle_points, sphere_angle):

    sched = []

    for (grid_width, grid_num) in zip(grid_widths, grid_nums):
        grid = b.utils.make_pose_grid_enumeration(
            -grid_width, -grid_width, -grid_width, rot_angle_bounds[0], 
            +grid_width, +grid_width, +grid_width, rot_angle_bounds[1],
            *grid_num,
            fib_points, angle_points, sphere_angle
        )
        sched.append(grid)
    return sched                                

def velocity_chm_builder(addresses, args):
    chm = genjax.choice_map({
                addresses[0][0]:genjax.index_choice_map(
                    jnp.arange(args[0].shape[0]),genjax.choice_map({
                        addresses[0][1]: args[0]
            }))
        })
    return chm

def unfold_with_proposals(T, proposal, unfold_vector):
    """
    Note that T starts from 1, where 0 is before the first run of 
    proposals: N x 4 x 4 of N proposed velocity vectors
    """
    return unfold_vector.at[T,...].set(proposal)

unfold_with_proposals_vmap = jax.jit(jax.vmap(unfold_with_proposals, in_axes = (None, 0, None)))
    

def c2f_pose_update_v1(trace_, key, t_arr, unfold_array, pose_grid, enumerator):
    
    T = jnp.argmax(t_arr)
    # N_prop x 100 x 4 x 4
    proposed_unfold_vectors = unfold_with_proposals_vmap(T, pose_grid, unfold_array)
    scores = enumerator.enumerate_choices_get_scores(trace_, key, proposed_unfold_vectors)
    return enumerator.update_choices(
        trace_, key,
        proposed_unfold_vectors[scores.argmax()]
    )
c2f_pose_update_v1_jit = jax.jit(c2f_pose_update_v1, static_argnames=("enumerator",))

def c2f_pose_update_v2(trace_, key, pose_grid, enumerator):
    
    scores = enumerator.enumerate_choices_get_scores(trace_, key, pose_grid)
    return enumerator.update_choices(
        trace_, key,
        pose_grid[scores.argmax()]
    )
c2f_pose_update_v2_jit = jax.jit(c2f_pose_update_v2, static_argnames=("enumerator",))