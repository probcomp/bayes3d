import jax
import jax.numpy as jnp


# # Run the `scorer_parallel` scorer over the batched proposals `proposals_batches`, on `gt_image`
# def batched_scorer_parallel(scorer_parallel, proposals_batches, gt_image, r):
#     def _score_batch(carry, proposals):  
#         # score over the selected rotation proposals
#         weights_new = scorer_parallel(proposals, gt_image, r)
#         # x, x_weight = proposals[jnp.argmax(weights_new)], jnp.max(weights_new)
#         # new_x, new_weight = jax.lax.cond(carry[-1] > jnp.max(weights_new), lambda: carry, lambda: (x, x_weight))

#         return 0, weights_new  # return highest weight pose proposal encountered so far
#     _, batched_weights = jax.lax.scan(_score_batch, 0, proposals_batches)
#     return batched_weights

def batched_scorer_parallel(scorer_parallel, proposals_batches, gt_image, r):
    def _score_batch(proposals):  
        weights_new = scorer_parallel(proposals, gt_image, r)
        return weights_new  # return highest weight pose proposal encountered so far
    score_batch_vmap = jax.vmap(_score_batch, in_axes=(0,))
    batched_weights = score_batch_vmap(proposals_batches)
    return batched_weights


