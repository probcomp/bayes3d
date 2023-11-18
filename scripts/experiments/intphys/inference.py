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

    enumerator = b.make_enumerator([("dynamics_1", "velocity")], chm_builder = velocity_chm_builder, argdiff_f=b.make_no_change_argdiffs)
    # then update trace over all the proposals
    velocity_vector = trace["dynamics_1", "velocity"]
    for grid in gridding_schedule:
        trace = c2f_pose_update_v1_jit(trace, key, jnp.zeros(2).at[1].set(1), velocity_vector, grid, enumerator)
    return trace

def inference_approach_D(model, gt, metadata):
    """
    2-step model with unfold
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
    enumerator = b.make_enumerator([("dynamics_1", "velocity")], chm_builder = velocity_chm_builder, argdiff_f=b.make_no_change_argdiffs)
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
            "init_pose" : metadata["init_pose"] # assume init pose is known
            }) 
        )
        if t > 1:
            chm = chm.unsafe_merge(genjax.choice_map(
            {"dynamics_1":genjax.index_choice_map(
                    jnp.arange(t-1),genjax.choice_map({
                        "velocity": velocity_vector[:t-1]
                }))}
            ))

        # RESORTING to model.importance as I am having issues with update and choicemaps with unfolds & maps
        _, trace = model.importance(key, chm, tuple(metadata["MODEL_ARGS"].values()))
        # put index number as one hot encoded
        t_arr = jnp.zeros(T+1).at[t].set(1)
        # then update trace over all the proposals

        for i, grid in enumerate(gridding_schedule):
            print("Grid #",i+1)
            trace = c2f_pose_update_v1_jit(trace, key, t_arr, velocity_vector, grid, enumerator)
    return trace

def inference_approach_E(model, gt, metadata):
    """
    2-step model with NO unfold
    This is NOT compatible with model_v1 but it is

    """
    T = metadata['MODEL_ARGS']['T_vec'].shape[0]

    # OR use 3d translation and rotation grid
    grid_widths = [0.2,0.1,0.05]
    grid_nums = [(3,3,3),(3,3,3),(3,3,3)]
    gridding_schedule = make_schedule_3d(grid_widths,grid_nums, [-jnp.pi/12, jnp.pi/12],10,10,jnp.pi/40)

    base_chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    # first make the chm builder:
    enumerators = [b.make_enumerator([(f"velocity_{i+1}")]) for i in range(T)]

    key = jax.random.PRNGKey(metadata["key_number"])
    # make initial sample:
    _, trace = model.importance(key, base_chm, tuple(metadata["MODEL_ARGS"].values()))

    for t in range(1,T+1):
        print("t = ", t)
        # force new constaints values to take over
        chm = base_chm.unsafe_merge(genjax.choice_map(
            {"depths" : genjax.index_choice_map(jnp.arange(t+1),genjax.choice_map({
                    "depths": gt[:t+1]
            })),
            "init_pose" : metadata["init_pose"], # assume init pose is known
            **{f"velocity_{i+1}" : trace[f"velocity_{i+1}"] for i in range(t-1)}
            }) 
        )

        # RESORTING to model.importance as I am having issues with update and choicemaps with unfolds &/or maps
        _, trace = model.importance(key, chm, tuple(metadata["MODEL_ARGS"].values()))

        # trace = trace.update(key, chm, b.make_unknown_change_argdiffs(trace))

        # then update trace over all the proposals
        for i, grid in enumerate(gridding_schedule):
            print("Grid #",i+1)
            trace = c2f_pose_update_v2_jit(trace, key, grid, enumerators[t-1])
    return trace

def inference_approach_F(model, gt, metadata):
    """
    2-step model with NO unfold
    HMM-style
    """
    # Use 3d translation and rotation grid
    grid_widths = [0.2,0.1,0.05, 0.025, 0.0125]
    grid_nums = [(3,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
    gridding_schedule = make_schedule_3d(grid_widths,grid_nums, [-jnp.pi/12, jnp.pi/12],10,10,jnp.pi/40)

    key = jax.random.PRNGKey(metadata["key_number"]+71)
    base_chm = genjax.choice_map(metadata["CHOICE_MAP_ARGS"])
    enumerator = b.make_enumerator(["velocity"])
    pose = metadata["INIT_POSE"]
    velocity = metadata["INIT_VELOCITY"]
    T = metadata["T"]
    traces = []
    model_args = metadata["MODEL_ARGS"]

    for t in range(1,T+1):
        print("t = ", t)
        # force new constaints values to take over
        chm = base_chm.unsafe_merge(genjax.choice_map({
            "depth" : gt[t]
        }))

        model_args["pose"] = pose
        model_args["velocity"] = velocity
        # RESORTING to model.importance as I am having issues with update and choicemaps with unfolds &/or maps
        _, trace = model.importance(key, chm, tuple(model_args.values()))

        # then update trace over all the proposals
        for i, grid in enumerate(gridding_schedule):
            # print("Grid #",i+1)
            trace = c2f_pose_update_v2_jit(trace, key, grid, enumerator)
        pose, velocity = trace.get_retval()[1]
        traces.append(trace)

    # first gt image can be assumed to be known as we have the init pose
    rendered = jnp.stack([gt[0]]+[tr.get_retval()[0] for tr in traces])
    return traces, rendered

def inference_approach_F2(model, gt, gridding_schedule, init_chm, T, model_args, init_state, key):
    """
    Sequential Importance Sampling on the unfolded HMM model
    with 'dumb' 3D pose enumeration proposal

    WITH JUST ONE PARTICLE
    """
    # extract data

    key, init_key = make_new_keys(key, 1)

    # define functions for SIS/SMC
    init_fn = jax.jit(model.importance)
    update_fn = jax.jit(model.update)
    proposal_fn = c2f_pose_update_v4_jit

    # initialize SMC/SIS
    init_log_weight, init_particle = init_fn(init_key, init_chm, (T, init_state, *model_args))
    _,_, init_particle, _ = model.update(init_key, init_particle, genjax.index_choice_map(
            [0], genjax.choice_map(
                {'velocity' : jnp.expand_dims(init_state[-1], axis = 0)}
            )),
            argdiffs_modelv5(init_particle, 0))

    argdiffs = argdiffs_modelv5(init_particle, 0)
    _, init_log_weight, init_particle, _ = update_fn(
            init_key, init_particle, update_choice_map(gt, 0), argdiffs)
    

    def smc_body(state, t):
        # get new keys
        print("jit compiling")
        jprint("t = {}",t)
        key, log_weight, particle = state
        key, update_key = make_new_keys(key, 1)
        key, proposal_key = make_new_keys(key, 1)

        argdiffs = argdiffs_modelv5(particle, t)

        # make enumerator for this time step (affects the proposal choice map)
        enumerator = b.make_enumerator([("velocity")], 
                                        chm_builder = proposal_choice_map,
                                        argdiff_f=lambda x: argdiffs,
                                        chm_args = [t])

        # update model to new depth observation
        _, update_log_weight, updated_particle, _ = update_fn(
            update_key, particle, update_choice_map(gt, t), argdiffs)

        # propose good poses based on proposal
        proposal_log_weight, new_particle = proposal_fn(
            proposal_key, updated_particle, gridding_schedule, enumerator, t)

        # get weight of particle
        new_log_weight = log_weight + proposal_log_weight + update_log_weight

        return (key, new_log_weight, new_particle), None

    (_, final_log_weight, particle), _ = jax.lax.scan(
        smc_body, (key, init_log_weight, init_particle), jnp.arange(1, T+1))
    print("SCAN finished")
    rendered = particle.get_retval()[0]
    return final_log_weight, particle, rendered
