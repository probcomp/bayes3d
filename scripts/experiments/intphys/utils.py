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


# Every thing below is from a working single particle tracker, which i will comment out for now
"""

def pose_update_v4(key, trace_, pose_grid, enumerator):
    
    weights = enumerator.enumerate_choices_get_scores(trace_, key, pose_grid)
    sampled_idx = weights.argmax() # jax.random.categorical(key, weights)

    return *enumerator.update_choices_with_weight(
        trace_, key,
        pose_grid[sampled_idx]
    ), pose_grid[sampled_idx]

pose_update_v4_jit = jax.jit(pose_update_v4, static_argnames=("enumerator",))


def c2f_pose_update_v4(key, trace_, gridding_schedule_stacked, enumerator, t):

    # reference_vel = jax.lax.cond(jnp.equal(t,1),lambda:trace_.args[1][1],lambda:trace_["velocity"][t-1])
    reference_vel = trace_["velocity"][t-1]
    for i in range(gridding_schedule_stacked.shape[0]):
        updated_grid = jnp.einsum("ij,ajk->aik", reference_vel, gridding_schedule_stacked[i])
        weight, trace_, reference_vel = pose_update_v4_jit(key, trace_, updated_grid, enumerator)
        
    return weight, trace_

c2f_pose_update_v4_vmap_jit = jax.jit(jax.vmap(c2f_pose_update_v4, in_axes=(0,0,None,None,None)),
                                    static_argnames=("enumerator", "t"))

c2f_pose_update_v4_jit = jax.jit(c2f_pose_update_v4,static_argnames=("enumerator", "t"))

def make_new_keys(key, N_keys):
    key, other_key = jax.random.split(key)
    new_keys = jax.random.split(other_key, N_keys)
    if N_keys > 1:
        return key, new_keys
    else:
        return key, new_keys[0]


def initial_choice_map(metadata):
    return genjax.index_choice_map(
            jnp.arange(0,metadata["T"]+1),
            genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
        )

def update_choice_map(gt, t):
    return genjax.index_choice_map(
            [t], genjax.choice_map(
                {'depth' : jnp.expand_dims(gt[t], axis = 0)}
            )
        )


def argdiffs_modelv5(trace, t):
"""
# Argdiffs specific to modelv5
"""
    # print(trace.args)
    args = trace.get_args()
    argdiffs = (
        Diff(args[0], NoChange),
        jtu.tree_map(lambda v: Diff(v, NoChange), args[1]),
        *jtu.tree_map(lambda v: Diff(v, NoChange), args[2:]),
    )
    return argdiffs

def proposal_choice_map(addresses, args, chm_args):
    addr = addresses[0] # custom defined
    return genjax.index_choice_map(
                    jnp.array([chm_args[0]]),genjax.choice_map({
                        addr: jnp.expand_dims(args[0], axis = 0)
            }))


"""