import collections
import heapq
import jax
import jax.numpy as jnp
import jax3dp3
import numpy as np

def c2f_contact_parameters(
    init_contact_parameters,
    contact_param_sched, 
    face_param_sched,
    likelihood_r_sched,
    contact_plane_pose,
    gt_image_masked,
    gt_img_complement,    
    model_box_dims,
    outlier_prob,
    outlier_volume,
    top_k=None
):
    """
    do coarse-to-fine, keeping the top top_k hypotheses at each round
    """
    num_steps = len(contact_param_sched)
    num_objects = model_box_dims.shape[0] 
    if top_k is None:
        top_k = num_objects
    top_k_heap = collections.deque([(float('-inf'), obj_idx, init_contact_parameters, None, None) for obj_idx in range(num_objects)])  # start with all objects inference
    
    for sched_i in range(num_steps):
    
        r = likelihood_r_sched[sched_i]
        contact_param_sweep_delta, face_param_sweep = contact_param_sched[sched_i], face_param_sched[sched_i]
                
        for _ in range(len(top_k_heap)):
            _, obj_idx, c, _, _ = top_k_heap.popleft()

            contact_param_sweep = contact_param_sweep_delta + c  # shift center 

            # get pose proposals in cam frame
            pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
                contact_param_sweep,
                face_param_sweep,
                model_box_dims[obj_idx],
                contact_plane_pose
            )  

            # get best pose proposal
            images_unmasked = jax3dp3.render_parallel(pose_proposals, obj_idx)
            images = jax3dp3.renderer.get_complement_masked_images(images_unmasked, gt_img_complement)
            weights = jax3dp3.threedp3_likelihood_parallel_jit(gt_image_masked, images, r, outlier_prob, outlier_volume)
            best_pose_idx = weights.argmax()

            top_k_heap.append((weights[best_pose_idx], obj_idx,
                               contact_param_sweep[best_pose_idx],
                               face_param_sweep[best_pose_idx],
                               pose_proposals[best_pose_idx]))  

        if sched_i == 0: 
            top_k_heap = collections.deque(heapq.nlargest(top_k, top_k_heap))  # after the first iteration prune search down to the top_k top hypothesis objects 
        
        # print(f"top {top_k} after {sched_i} iters:\n: {[(s, model_names[i]) for (s,i,p) in top_k_heap]}")

    items = [item for item in heapq.nlargest(top_k, top_k_heap)]
    scores = np.array([item[0] for item in items])
    items = [items[i] for i in np.argsort(-scores)]
    return items


def multi_panel_c2f(results:list, r, gt_img_complement, gt_image_masked, rgb_viz, h, w, max_depth, outlier_prob, outlier_volume, model_names,title=None):
    overlays = []
    labels = []
    scores = []
    imgs = []

    scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))
    rgb_viz_resized = jax3dp3.viz.resize_image(rgb_viz,h,w)
    for i in range(len(results)):
        gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_masked[:,:,2],  max=max_depth)

        score_orig, obj_idx, _, _, pose = results[i]
        image_unmasked = jax3dp3.render_single_object(pose, obj_idx)
        image = jax3dp3.renderer.get_complement_masked_image(image_unmasked, gt_img_complement)
        imgs.append(image)

        score = scorer_parallel_jit(gt_image_masked, image[None, ...], r, outlier_prob, outlier_volume)[0]

        overlays.append(
            jax3dp3.viz.overlay_image(rgb_viz_resized, jax3dp3.viz.get_depth_image(image_unmasked[:,:,2],  max=max_depth))
        )
        scores.append(score)
        labels.append(
            "Obj {:d}: {:s}\n Score Orig: {:.2f} \n Score: {:.2f}".format(obj_idx, model_names[obj_idx], score_orig, score)
        )

    normalized_probabilites = jax3dp3.utils.normalize_log_scores(jnp.array(scores))

    dst = jax3dp3.viz.multi_panel(
        [rgb_viz_resized, gt_img_viz, *overlays],
        labels=["RGB", "Depth Segment", *labels],
        bottom_text="{}\n Normalized Probabilites: {}".format(jnp.array(scores), jnp.round(normalized_probabilites, decimals=4)),
        label_fontsize =15,
        title=title
    )

    return dst