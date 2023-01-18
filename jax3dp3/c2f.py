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


def multi_panel_gt_images(rgb_data, gt_image_full_data, gt_image_above_table_data, segmentation_image_data, h, w, max_depth, title="GT Images", img_dir="imgs",scene_id=0):
    imgs = []  # add images of dim h-by-w
    labels = []
    scores = []
    imgs = []
    
    # add original RGB image
    rgb_viz = jax3dp3.viz.get_rgb_image(rgb_data, 255.0)
    rgb_viz = jax3dp3.viz.resize_image(rgb_viz, h,w)
    imgs.append(rgb_viz)
    labels.append("GT RGB Image (full)")
    rgb_viz.save(f"{img_dir}/rgb_{scene_id}.png")

    # add original GT depth image
    # gt_image_data = jax3dp3.transforms_3d.depth_to_point_cloud_image(depth_data, fx,fy,cx,cy)
    gt_image_full = jax3dp3.viz.get_depth_image(gt_image_full_data[:,:,2], max=max_depth)
    imgs.append(gt_image_full)
    labels.append("GT Depth Image (full)")
    gt_image_full.save(f"{img_dir}/gt_image_full_{scene_id}.png")

    # add above-table GT image
    # gt_point_cloud_full = jax3dp3.transforms_3d.point_cloud_image_to_points(gt_image_data)
    # table_pose, _ = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < max_depth, :], 
    # ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004)
    # gt_image_above_table_data = gt_image_full * (jax3dp3.transforms_3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.02)[:,:,None]
    gt_image_above_table_viz = jax3dp3.viz.get_depth_image(gt_image_above_table_data[:,:,2], max=max_depth)
    imgs.append(gt_image_above_table_viz)
    labels.append("GT Depth Image (above table)")
    gt_image_above_table_viz.save(f"{img_dir}/gt_image_above_table_{scene_id}.png")

    # add segmented depth image
    segmentation_image_viz = jax3dp3.viz.get_depth_image(segmentation_image_data + 1, max=segmentation_image_data.max() + 1)
    imgs.append(segmentation_image_viz)
    labels.append("Segmentation Results")
    segmentation_image_viz.save(f"{img_dir}/segmentation_image_{scene_id}.png")

    # add segmentation-overlaid GT image
    segmentation_on_gt_viz = jax3dp3.viz.overlay_image(rgb_viz, segmentation_image_viz)
    imgs.append(segmentation_on_gt_viz)
    labels.append("Segmentation overlap on GT")

    dst = jax3dp3.viz.multi_panel(
        [*imgs],
        labels=[*labels],
        label_fontsize =15,
        title=title
    )

    return dst



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