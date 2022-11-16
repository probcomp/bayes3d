import jax
import jax.numpy as jnp

def batch_split(proposals, num_batches):
    num_proposals = proposals.shape[0]
    if num_proposals % num_batches != 0:
        print(f"WARNING: {num_proposals} Not evenly divisible by {num_batches}; defaulting to 2x split")  # TODO find a good factor
        num_batches = 2
    return jnp.array(jnp.split(proposals, num_batches))

# Run the `scorer_parallel` scorer over the batched proposals `proposals_batches`, on `gt_image`
def batched_scorer_parallel(scorer_parallel, num_batches, proposals, gt_image, r):
    def _score_batch(carry, proposals):  
        # score over the selected rotation proposals
        weights_new = scorer_parallel(proposals, gt_image, r)
        return 0, weights_new  # return highest weight pose proposal encountered so far
    proposals_batches = batch_split(proposals, num_batches)
    _, batched_weights = jax.lax.scan(_score_batch, 0, proposals_batches)

    return batched_weights.ravel()

# # TODO: take non-batched input

# def batched_scorer_parallel(scorer_parallel, proposals_batches, gt_image, r):
#     def _score_batch(proposals):  
#         weights_new = scorer_parallel(proposals, gt_image, r)
#         return weights_new  # return highest weight pose proposal encountered so far
#     score_batch_vmap = jax.vmap(_score_batch, in_axes=(0,))
#     batched_weights = score_batch_vmap(proposals_batches)
#     return batched_weights


