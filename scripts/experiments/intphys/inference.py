import genjax
import bayes3d as b
import jax
import jax.numpy as jnp
import numpy as np
from genjax.inference.importance_sampling import importance_sampling, sampling_importance_resampling

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

