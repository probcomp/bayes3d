"""
just saving old pose update code here. This file is NOT meant to be loaded/imported
"""

def pose_likelihood_looper(proposals, scores):
    jprint("Max Score: {}", jnp.max(scores))
    jax.lax.fori_loop(0, proposals.shape[0],
    pose_likelihood_printer, (proposals, scores))

def pose_likelihood_printer(i, inp):
    proposals, scores = inp
    jprint("{}: {}", cam_pose @ proposals[i,:,3], scores[i])
    return proposals, scores

def update_pose_estimate(memory, gt_image):

    old_pose_estimate, prev_pose, T = memory

    threedp3_weight = 1
    proposals = jnp.einsum("ij,ajk->aik", old_pose_estimate, translation_deltas)
    rendered_images = b.RENDERER.render_multiobject_parallel(jnp.stack([proposals, occ_poses_trans]), jnp.array([0,1]))

    threedp3_scores = threedp3_weight * b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.001, 0.1, 10**3, 3)
    unique_best_3dp3_score = jnp.sum(threedp3_scores == threedp3_scores[jnp.argmax(threedp3_scores)]) == 1

    physics_weight = jax.lax.cond(unique_best_3dp3_score, lambda _ : 0, lambda _ : 10000, None)

    physics_estimated_pose = p.physics_prior_v1_jit(old_pose_estimate, prev_pose, jnp.array([1,1,1]), cam_pose, world2cam)

    physics_scores = jax.lax.cond(jnp.greater(T, 1), 
    lambda _ : physics_weight * p.physics_prior_parallel_jit(proposals, physics_estimated_pose), 
    lambda _ : jnp.zeros(threedp3_scores.shape[0]), 
    None)

    scores = threedp3_scores + physics_scores

    pose_estimate = proposals[jnp.argmax(scores)]

    # pose_world = cam_pose @ pose_estimate
    # jprint("{}: {}, {}", T, unique_best_3dp3_score, pose_world[:,3])

    # proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    # rendered_images = b.RENDERER.render_multiobject_parallel(jnp.stack([proposals, occ_poses_rot]), jnp.array([0,1]))
    # weights_new = b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    # pose_estimate = proposals[jnp.argmax(weights_new)]

    return (pose_estimate, old_pose_estimate, T+1), pose_estimate

def update_pose_estimate_c2f(pose_estimate, gt_image):
    for translation_deltas in gridding:
        proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
        rendered_images = b.RENDERER.render_multiobject_parallel(jnp.stack([proposals, occ_poses_trans]), jnp.array([0,1]))
        weights_new = b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
        pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    rendered_images = b.RENDERER.render_multiobject_parallel(jnp.stack([proposals, occ_poses_rot]), jnp.array([0,1]))
    weights_new = b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate