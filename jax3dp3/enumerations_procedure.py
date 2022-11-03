import jax
import jax.numpy as jnp

def enumerative_inference_single_frame(scorer_parallel, gt_image, proposals_batches):  
    # scan over batches of rotation proposals for single image
    def _enum_infer_batch_scan(carry, proposals):   
        # score over the selected rotation proposals
        weights_new = scorer_parallel(proposals, gt_image)
        x, x_weight = proposals[jnp.argmax(weights_new)], jnp.max(weights_new)
        new_x, new_weight = jax.lax.cond(carry[-1] > jnp.max(weights_new), lambda: carry, lambda: (x, x_weight))

        return (new_x, new_weight), None  # return highest weight pose proposal encountered so far
    best_prop, _ = jax.lax.scan(_enum_infer_batch_scan, (jnp.empty((4,4)), jnp.NINF), proposals_batches)
    return best_prop


#### 
# Experimental
####

def inference_frame_joint_nested(scorer_parallel, gt_image, translation_proposals, rotation_proposals_batches):  # scan over batches of rotation proposals for single image
    def _enum_rotation_batch_scan(carry, proposals):  # enumerate over rotations
        # score over the selected rotation proposals
        weights_new = scorer_parallel(proposals, gt_image)
        x, x_weight = proposals[jnp.argmax(weights_new)], jnp.max(weights_new)

        # prev_x, prev_weight = carry
        new_x, new_weight = jax.lax.cond(carry[-1] > jnp.max(weights_new), lambda: carry, lambda: (x, x_weight))

        return (new_x, new_weight), None  # return highest weight pose proposal encountered so far
    
    def _enum_translation_batch(translation_proposal):  # enumerate over translations
        best_pose_weight, _ = jax.lax.scan(_enum_rotation_batch_scan, (jnp.empty((4,4)), jnp.NINF), jnp.einsum("ij,abjk->abik", translation_proposal, rotation_proposals_batches))
        return best_pose_weight  # best rotation for the given translation

    enum_translation_batch_vmap = jax.vmap(_enum_translation_batch, (0,))
    best_pose, best_weights = enum_translation_batch_vmap(translation_proposals)
    best_pose = best_pose[jnp.argmax(best_weights)]

    return best_pose