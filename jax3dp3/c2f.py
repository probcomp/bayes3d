import collections
import heapq
import jax
import jax.numpy as jnp
import jax3dp3
import numpy as np



def make_schedules(grid_widths, grid_params):
    contact_param_sched = []
    face_param_sched = []
    for (grid_width, grid_param) in zip(grid_widths, grid_params):
        c, f = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
            -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi*2, 
            *grid_param,
            jnp.arange(6)
        )
        contact_param_sched.append(c)
        face_param_sched.append(f)
    return contact_param_sched, face_param_sched

def c2f_contact_parameters(
    init_contact_parameters,
    contact_param_sched, 
    face_param_sched,
    likelihood_r_sched,
    r_overlap_check,
    r_final,
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

    final_results = []
    for i in range(len(items)):
        score_orig, obj_idx, _, _, pose = items[i]
        image_unmasked = jax3dp3.render_single_object(pose, obj_idx)
        image = jax3dp3.renderer.get_complement_masked_image(image_unmasked, gt_img_complement)
        
        overlap = jax3dp3.likelihood.threedp3_likelihood_get_counts(gt_image_masked, image, r_overlap_check)
        score = jax3dp3.threedp3_likelihood_parallel_jit(gt_image_masked, image[None, ...], r_final, outlier_prob, outlier_volume)[0]

        final_results.append((score, overlap, pose, obj_idx, image, image_unmasked))

    normalized_probabilites = jax3dp3.utils.normalize_log_scores(jnp.array([item[0] for item in final_results]))
    final_results = [(*data, prob) for (data,prob) in zip(final_results, normalized_probabilites)]
    return final_results


def multi_panel_c2f_viz(results:list, rgb, point_cloud_image, h, w, max_depth, model_names,title=None):
    if rgb.shape[-1] == 3:
        rgb_viz = jax3dp3.viz.get_rgb_image(rgb, 255.0)
    else:
        rgb_viz = jax3dp3.viz.get_rgba_image(rgb, 255.0)

    rgb_viz_resized = jax3dp3.viz.resize_image(rgb_viz,h,w)
    gt_img_viz = jax3dp3.viz.get_depth_image(point_cloud_image[:,:,2],  max=max_depth)

    overlays = []
    scores = []
    labels = []
    probabilities = []
    for i in range(len(results)):
        (score, overlap, pose, obj_idx, image, image_unmasked, prob) = results[i]
        overlays.append(
            jax3dp3.viz.overlay_image(rgb_viz_resized, jax3dp3.viz.get_depth_image(image_unmasked[:,:,2],  max=max_depth))
        )
        scores.append(score)
        labels.append(
            "Obj {:d}: {:s}\n Score: {:.2f}".format(obj_idx, model_names[obj_idx], score)
        )
        probabilities.append(prob)

    overlap = results[0][1]
    dst = jax3dp3.viz.multi_panel(
        [rgb_viz_resized, gt_img_viz, *overlays],
        labels=["RGB", "Depth Segment", *labels],
        bottom_text="{}\n Normalized Probabilites: {}\n Overlap: {}".format(
            jnp.array(scores), jnp.round(jnp.array(probabilities), decimals=4),
            "{:.4f} {:.4f}".format(overlap[0] / overlap[1], overlap[2] / overlap[3])
        ),
        label_fontsize =15,
        title=title
    )

    return dst