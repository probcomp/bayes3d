import genjax
import bayes3d as b
import jax
import jax.numpy as jnp
import numpy as np
from genjax.inference.importance_sampling import importance_sampling, sampling_importance_resampling
from utils import *

def inference_approach_A(model, gt, metadata, num_particles):
    """
    IS + MLE: Get a bunch of importance samples and use MLE
    Over the FULL T timesteps
    """
    chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    # force new constaints values to take over
    chm = chm.unsafe_merge(genjax.choice_map(
        {"depths" : genjax.vector_choice_map(genjax.choice_map({
                "depths": gt
        }))}))
    
    key = jax.random.PRNGKey(metadata["key_number"])
    # subkeys = jax.random.split(key, num)
    imp = importance_sampling(
        model, num_particles)
    (trs, lnw, lmle, lws) = imp.apply(
        key, chm, tuple(metadata["MODEL_ARGS"].values()))
    
    tr = jax.tree_util.tree_map(lambda v: v[jnp.argmax(lnw)], trs)
    return tr


def inference_approach_B(model, gt, metadata, num_particles):
    """
    SIR: Get a bunch of importance samples and sample using their weights
    Over the FULL T timesteps
    """
    chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    # force new constaints values to take over
    chm = chm.unsafe_merge(genjax.choice_map(
        {"depths" : genjax.vector_choice_map(genjax.choice_map({
                "depths": gt
        }))}))
    
    key = jax.random.PRNGKey(metadata["key_number"])
    # subkeys = jax.random.split(key, num)
    imp = sampling_importance_resampling(
        model, num_particles)
    (tr, lnw, log_ml_estimate) = imp.apply(
        key, chm, tuple(metadata["MODEL_ARGS"].values()))
    return tr

def inference_approach_C(model, gt, metadata):
    """
    Greedy Grid Enumeration of T=0 to T=1
    """
    chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    # force new constaints values to take over
    chm = chm.unsafe_merge(genjax.choice_map(
        {"depths" : genjax.vector_choice_map(genjax.choice_map({
                "depths": gt
        })),
        "init_pose" : metadata["init_pose"] # assume init pose is known
        }) 
    )
    
    # make 3d translation grid: list of N x 4 x 4 poses
    grid_widths = [0.1,0.05,0.025]
    grid_nums = [(3,3,3),(3,3,3),(3,3,3)]
    gridding_schedule = make_schedule_translation_3d(grid_widths, grid_nums)

    # make initial sample:
    key = jax.random.PRNGKey(metadata["key_number"])
    _, trace = model.importance(key, chm, tuple(metadata["MODEL_ARGS"].values()))
    # return trace

    # do inference by updating the T=1 slice of the velocity address
    # first make the chm builder:

    enumerator = b.make_enumerator([("dynamics_1", "velocity")], chm_builder = velocity_chm_builder)
    # then update trace over all the proposals
    velocity_vector = trace["dynamics_1", "velocity"]
    for grid in gridding_schedule:
        trace = c2f_pose_update_jit(trace, key, jnp.zeros(2).at[1].set(1), velocity_vector, grid, enumerator)
    return trace

def inference_approach_D(model, gt, metadata):
    """
    Greedy Grid Enumeration of T=0 to T=5
    # I think this might just be the 2-step model with the unfold structure
    """
    T = metadata['MODEL_ARGS']['T_vec'].shape[0]
    # make 3d translation grid: list of N x 4 x 4 poses
    # grid_widths = [0.1,0.05,0.025]
    # grid_nums = [(3,3,3),(3,3,3),(3,3,3)]
    # gridding_schedule = make_schedule_translation_3d(grid_widths, grid_nums)

    # OR use 3d translation and rotation grid
    grid_widths = [0.2,0.1,0.05]
    grid_nums = [(3,3,3),(3,3,3),(3,3,3)]
    # sched = make_schedule_3d([0.1],[(1,1,1)], [-jnp.pi/6, jnp.pi/6],50,10,jnp.pi/40)
    gridding_schedule = make_schedule_3d(grid_widths,grid_nums, [-jnp.pi/12, jnp.pi/12],10,10,jnp.pi/40)

    base_chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    # first make the chm builder:
    enumerator = b.make_enumerator([("dynamics_1", "velocity")], chm_builder = velocity_chm_builder)
    key = jax.random.PRNGKey(metadata["key_number"])
    # make initial sample:
    _, trace = model.importance(key, base_chm, tuple(metadata["MODEL_ARGS"].values()))

    for t in range(1,T+1):
        print("t = ", t)
        velocity_vector = trace["dynamics_1", "velocity"]
        # force new constaints values to take over
        chm = base_chm.unsafe_merge(genjax.choice_map(
            {"depths" : genjax.index_choice_map(jnp.arange(t+1),genjax.choice_map({
                    "depths": gt[:t+1]
            })),
            "init_pose" : metadata["init_pose"], # assume init pose is known
            "dynamics_1":genjax.index_choice_map(
                jnp.arange(velocity_vector.shape[0]),genjax.choice_map({
                    "velocity": velocity_vector
            }))
            }) 
        )

        # RESORTING to model.importance as I am having issues with update and choicemaps with unfolds & maps
        _, trace = model.importance(key, chm, tuple(metadata["MODEL_ARGS"].values()))
        # put index number as one hot encoded
        t_arr = jnp.zeros(T+1).at[t].set(1)
        # then update trace over all the proposals

        for i, grid in enumerate(gridding_schedule):
            print("Grid #",i+1)
            trace = c2f_pose_update_jit(trace, key, t_arr, velocity_vector, grid, enumerator)
    return trace