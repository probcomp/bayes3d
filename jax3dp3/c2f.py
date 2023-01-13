import collections
import heapq
import jax3dp3

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
    return items