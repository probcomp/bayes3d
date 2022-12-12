import jax
from jax3dp3.batched_scorer import batched_scorer_parallel
import jax.numpy as jnp

def coarse_to_fine(scorer, enum_grid_tr, enum_grid_r, enum_likelihood_r, latent_pose_estimate, gt_image):
    NUM_BATCHES = 4
    for cnt in range(len(enum_likelihood_r)):
        r = enum_likelihood_r[cnt]
        enumerations_t =  enum_grid_tr[cnt]
        proposals = jnp.einsum("...ij,...jk->...ik", latent_pose_estimate, enumerations_t, optimize="optimal")

        scorer_partial = lambda pose: scorer(pose, gt_image, r)
        scorer_parallel = jax.vmap(scorer_partial, in_axes = (0, ))

        # translation inference
        weights = batched_scorer_parallel(scorer_parallel, NUM_BATCHES, proposals) 
        best_pose_estimate = proposals[jnp.argmax(weights)]

        # rotation inference
        enumerations_r = enum_grid_r[cnt]
        proposals = jnp.einsum('ij,ajk->aik', best_pose_estimate, enumerations_r, optimize="optimal")

        weights = batched_scorer_parallel(scorer_parallel, NUM_BATCHES, proposals) 
        best_pose_estimate = proposals[jnp.argmax(weights)]

        latent_pose_estimate = best_pose_estimate


    return best_pose_estimate


def coarse_to_fine_pose_and_weights(scorer, enum_grid_tr, enum_grid_r, enum_likelihood_r, latent_pose_estimate, gt_image):
    NUM_BATCHES = 4
    for cnt in range(len(enum_likelihood_r)):
        r = enum_likelihood_r[cnt]
        enumerations_t =  enum_grid_tr[cnt]
        proposals = jnp.einsum("...ij,...jk->...ik", latent_pose_estimate, enumerations_t, optimize="optimal")

        scorer_partial = lambda pose: scorer(pose, gt_image, r)
        scorer_parallel = jax.vmap(scorer_partial, in_axes = (0, ))

        # translation inference
        tr_weights = batched_scorer_parallel(scorer_parallel, NUM_BATCHES, proposals) 
        best_pose_estimate = proposals[jnp.argmax(tr_weights)]

        # rotation inference
        enumerations_r = enum_grid_r[cnt]
        proposals = jnp.einsum('ij,ajk->aik', best_pose_estimate, enumerations_r, optimize="optimal")

        rot_weights = batched_scorer_parallel(scorer_parallel, NUM_BATCHES, proposals) 
        best_pose_estimate = proposals[jnp.argmax(rot_weights)]

        latent_pose_estimate = best_pose_estimate
    best_pose_weight = rot_weights[jnp.argmax(rot_weights)]


    return jnp.argmax(tr_weights), jnp.argmax(rot_weights), best_pose_weight