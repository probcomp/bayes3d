import collections
import heapq
import jax
import jax.numpy as jnp
import jax3dp3
import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import numpy as np

####################
# Scheduling functions
####################

def make_schedules_contact_params(grid_widths, rotation_angle_widths, grid_params):
    ## version of make_schedules with angle range reduction based on previous iter
    sched = []

    for (grid_width, angle_width, grid_param) in zip(grid_widths, rotation_angle_widths, grid_params):
        cf = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
            -grid_width, -grid_width, -angle_width, 
            +grid_width, +grid_width, angle_width, 
            *grid_param,  # *grid_param is num_x, num_y, num_angle
            jnp.arange(6)
        )
        sched.append(cf)
    return sched

def make_schedules_full_pose_params(grid_widths, sphere_angle_widths, rotation_angle_widths, grid_params):
    ## version of make_schedules with angle range reduction based on previous iter
    sched = [] 
    for (grid_width, sphere_angle_width, rotation_angle_width, grid_param) in zip(grid_widths, sphere_angle_widths, rotation_angle_widths, grid_params):
        min_x, min_y, min_z = -grid_width, -grid_width, -grid_width
        max_x,max_y,max_z = grid_width, grid_width, grid_width
        num_x, num_y, num_z, num_fib_sphere, num_planar_angle = grid_param

        cf = jax3dp3.enumerations.make_grid_enumeration(
            min_x, min_y, min_z, -rotation_angle_width,
            max_x,max_y,max_z, rotation_angle_width, 
            num_x,num_y,num_z, num_fib_sphere, num_planar_angle, 
            sphere_angle_width)
        sched.append(cf)
    return sched

def make_schedules(grid_widths, angle_widths, grid_params, full_pose=False, sphere_angle_widths=None):
    # check schedule validity
    assert len(grid_widths) == len(angle_widths) == len(grid_params)  
    
    if not full_pose: 
        assert len(grid_params[0]) == 3, "pass in (num_x, num_y, num_angles) as grid_param"
        return make_schedules_contact_params(grid_widths, angle_widths, grid_params)   
    else:
        assert len(grid_params[0]) == 5, "pass in (num_x, num_y, num_z, num_fib_sphere, num_planar_angle) as grid_param"
        if sphere_angle_widths is None: 
            sphere_angle_widths = [jnp.pi for _ in angle_widths]
        assert len(grid_widths) == len(sphere_angle_widths)
        return make_schedules_full_pose_params(grid_widths, sphere_angle_widths, angle_widths, grid_params)

################
# Scoring functions
################
def batch_split(proposals, num_batches):
    num_proposals = proposals.shape[0]
    if num_proposals % num_batches != 0:
        # print(f"WARNING: {num_proposals} Not evenly divisible by {num_batches}; defaulting to 3x split")  # TODO find a good factor
        num_batches = 3
    return jnp.array(jnp.split(proposals, num_batches))

def score_poses_batched(
    renderer,
    obj_idx,
    obs_point_cloud_image,
    obs_point_cloud_image_complement,
    pose_proposals,
    r_sweep,
    outlier_prob,
    outlier_volume,
    num_batches 
):
    # get best pose proposal

    proposals_batches = batch_split(pose_proposals, num_batches)
    
    print("batch split complete")
    fully_occluded_weights = jax3dp3.threedp3_likelihood_jit(
        obs_point_cloud_image, obs_point_cloud_image, r_sweep[0], outlier_prob, outlier_volume
    )
    
    print("proposals complete")
    all_weights = None
    for pose_proposals_batch in proposals_batches:
        rendered_object_images = renderer.render_parallel(pose_proposals_batch, obj_idx)[...,:3]
        rendered_images = jax3dp3.splice_image_parallel(rendered_object_images, obs_point_cloud_image_complement)
    
        print("looping through batch")

        weights = jax3dp3.threedp3_likelihood_with_r_parallel_jit(
            obs_point_cloud_image, rendered_images, r_sweep, outlier_prob, outlier_volume
        )

        if all_weights is None:
            all_weights = weights 
        else:
            all_weights = jnp.append(all_weights, weights, axis=1)

    # return weights, fully_occluded_weight
    return all_weights, fully_occluded_weights

def score_contact_parameters(
    renderer,
    obj_idx,
    obs_point_cloud_image,
    obs_point_cloud_image_complement,
    sweep,
    contact_plane_pose_in_cam_frame,
    r_sweep,
    outlier_prob,
    outlier_volume,
    model_box_dims,
    num_batches 
):
    contact_param_sweep, face_param_sweep = sweep
    pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
        contact_param_sweep,
        face_param_sweep,
        model_box_dims[obj_idx],
        contact_plane_pose_in_cam_frame
    )

    # get best pose proposal
    weights, fully_occluded_weight = score_poses_batched(
                                        renderer,
                                        obj_idx,
                                        obs_point_cloud_image,
                                        obs_point_cloud_image_complement,
                                        pose_proposals,
                                        r_sweep,
                                        outlier_prob,
                                        outlier_volume,
                                        num_batches
                                    )
    return pose_proposals, weights, fully_occluded_weight

def get_probabilites(hypotheses):
    scores = jnp.array( [i[0] for i in hypotheses])
    normalized_scores = jax3dp3.utils.normalize_log_scores(scores)
    return normalized_scores


#################
# c2f main functions
#################

def c2f_contact_parameters(
    renderer,
    obs_point_cloud_image,
    obs_point_cloud_image_masked,
    obs_point_cloud_image_complement,
    sched,
    contact_plane_pose_in_cam_frame,
    r_sweep,
    outlier_prob,
    outlier_volume,
    model_box_dims,
    obj_idx_hypotheses=None,
    num_batches=100
):

    masked_cloud = obs_point_cloud_image_masked.reshape(-1, 3)
    masked_cloud = masked_cloud[masked_cloud[:,2] < renderer.intrinsics.far,:]
    points_in_table_ref_frame =  t3d.apply_transform(
        masked_cloud,
        t3d.inverse_pose(contact_plane_pose_in_cam_frame)
    )
    point_seg = j.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
    points_filtered = points_in_table_ref_frame[point_seg == j.utils.get_largest_cluster_id_from_segmentation(point_seg)]
    center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
    
    contact_init = jnp.array([center_x, center_y, 0.0])

    hypotheses = []
    if obj_idx_hypotheses is None:
        obj_idx_hypotheses = jnp.arange(len(renderer.meshes))
    for obj_idx in obj_idx_hypotheses:
        hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

    hypotheses_over_time = [hypotheses]


    for c2f_iter in range(len(sched)):
        contact_param_sweep_delta, face_param_sweep = sched[c2f_iter]
        new_hypotheses = []

        for hyp in hypotheses:
            old_score = hyp[0]
            obj_idx = hyp[1]
            contact_params = hyp[2]

            new_contact_param_sweep = contact_params + contact_param_sweep_delta
            
            pose_proposals, weights, _ = score_contact_parameters(
                renderer,
                obj_idx,
                obs_point_cloud_image,
                obs_point_cloud_image_complement,
                (new_contact_param_sweep, face_param_sweep),
                contact_plane_pose_in_cam_frame,
                r_sweep,
                outlier_prob,
                outlier_volume,
                model_box_dims,
                num_batches 
            )
            r_idx, best_idx = jnp.unravel_index(weights.argmax(), weights.shape)

            new_hypotheses.append(
                (
                    weights[r_idx, best_idx],
                    obj_idx,
                    new_contact_param_sweep[best_idx],
                    pose_proposals[best_idx]
                )
            )

        hypotheses_over_time.append(new_hypotheses)
        hypotheses = new_hypotheses
    return hypotheses_over_time

def c2f_full_pose(
    renderer,
    obs_point_cloud_image,
    obs_point_cloud_image_masked,
    obs_point_cloud_image_complement,
    sched,
    r_sweep,
    outlier_prob,
    outlier_volume,
    obj_idx_hypotheses=None,
    num_batches=100
):
    print("Entering full pose-only coarse to fine")

    masked_cloud = obs_point_cloud_image_masked.reshape(-1, 3)
    masked_cloud = masked_cloud[masked_cloud[:,2] < renderer.intrinsics.far,:]
    # points_in_table_ref_frame =  t3d.apply_transform(
    #     masked_cloud,
    #     t3d.inverse_pose(contact_plane_pose_in_cam_frame)
    # )
    point_seg = j.utils.segment_point_cloud(masked_cloud, 0.1)
    points_filtered = masked_cloud[point_seg == j.utils.get_largest_cluster_id_from_segmentation(point_seg)]
    center_x, center_y, center_z = ( points_filtered.min(0) + points_filtered.max(0))/2
    
    pose_init = t3d.transform_from_pos(jnp.array([center_x, center_y, center_z]))

    # Setup object id hypotheses; default is every model loaded into the renderer
    hypotheses = []
    if obj_idx_hypotheses is None:  
        obj_idx_hypotheses = jnp.arange(len(renderer.meshes))
    for obj_idx in obj_idx_hypotheses:
        hypotheses += [(-jnp.inf, obj_idx, pose_init)]
    hypotheses_over_time = []

    for c2f_iter in range(len(sched)):
        pose_sweep_delta = sched[c2f_iter]
        new_hypotheses = []

        for hyp in hypotheses:
            old_score = hyp[0]
            obj_idx = hyp[1]
            pose_hyp = hyp[2]
            new_pose_proposals = jnp.einsum('ij,bjk->bik', pose_hyp, pose_sweep_delta)
            
            weights, _ = score_poses_batched(
                                    renderer,
                                    obj_idx,
                                    obs_point_cloud_image,
                                    obs_point_cloud_image_complement,
                                    new_pose_proposals,
                                    r_sweep,
                                    outlier_prob,
                                    outlier_volume,
                                    num_batches
                                )

            r_idx, best_idx = jnp.unravel_index(weights.argmax(), weights.shape)

            new_hypotheses.append(
                (
                    weights[r_idx, best_idx],
                    obj_idx,
                    new_pose_proposals[best_idx],
                )
            )

        hypotheses_over_time.append(new_hypotheses)
        hypotheses = new_hypotheses
    return hypotheses_over_time

def c2f_occlusion_viz(
    good_poses,
    pose_proposals,
    ranked_high_value_seg_ids,
    rgb_original,
    obj_idx,
    obs_point_cloud_image,
    segmentation_image,
    camera_params
):
    (h,w,fx,fy,cx,cy, near, far) = camera_params
    rgb_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original, 255.0), h,w)
    all_images_overlayed = jax3dp3.render_multiobject(pose_proposals, [obj_idx for _ in range(pose_proposals.shape[0])])
    enumeration_viz_all = jax3dp3.viz.get_depth_image(all_images_overlayed[:,:,2], max=far)

    all_images_overlayed = jax3dp3.render_multiobject(good_poses, [obj_idx for _ in range(good_poses.shape[0])])
    enumeration_viz = jax3dp3.viz.get_depth_image(all_images_overlayed[:,:,2], max=far)
    overlay_viz = jax3dp3.viz.overlay_image(rgb_viz, enumeration_viz)
    
    if len(ranked_high_value_seg_ids) > 0:
        image_masked = jax3dp3.get_image_masked_and_complement(
            obs_point_cloud_image, segmentation_image, ranked_high_value_seg_ids[0], far
        )[0]
        best_occluder = jax3dp3.viz.get_depth_image(image_masked[:,:,2],  max=far)
    else:
        best_occluder = jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2],  max=far)

    return jax3dp3.viz.multi_panel(
        [rgb_viz, overlay_viz, best_occluder],
        labels=["RGB", "Pose Distribution", "Likely Occluder"],
    )
