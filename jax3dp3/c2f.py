import collections
import heapq
import jax
import jax.numpy as jnp
import jax3dp3
import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import numpy as np

def make_schedules(grid_widths, angle_widths, grid_params):
    ## version of make_schedules with angle range reduction based on previous iter
    contact_param_sched = []
    face_param_sched = []

    for (grid_width, angle_width, grid_param) in zip(grid_widths, angle_widths, grid_params):
        c, f = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
            -grid_width, -grid_width, -angle_width, +grid_width, +grid_width, angle_width, 
            *grid_param,
            jnp.arange(6)
        )
        contact_param_sched.append(c)
        face_param_sched.append(f)
    return (contact_param_sched, face_param_sched)


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
):
    contact_param_sched, face_param_sched = sched

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
    for obj_idx in jnp.arange(len(renderer.meshes)):
        hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

    hypotheses_over_time = [hypotheses]


    for c2f_iter in range(len(contact_param_sched)):
        contact_param_sweep_delta, face_param_sweep = contact_param_sched[c2f_iter], face_param_sched[c2f_iter]
        new_hypotheses = []

        for hyp in hypotheses:
            old_score = hyp[0]
            obj_idx = hyp[1]
            contact_params = hyp[2]

            new_contact_param_sweep = contact_params + contact_param_sweep_delta
            
            pose_proposals, weights, _ = c2f_score_contact_parameters(
                renderer,
                obj_idx,
                obs_point_cloud_image,
                obs_point_cloud_image_complement,
                new_contact_param_sweep,
                face_param_sweep,
                r_sweep,
                contact_plane_pose_in_cam_frame,
                outlier_prob,
                outlier_volume,
                model_box_dims,
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



###### Occlusion check
def c2f_score_contact_parameters(
    renderer,
    obj_idx,
    obs_point_cloud_image,
    obs_point_cloud_image_complement,
    contact_param_sweep,
    face_param_sweep,
    r_sweep,
    contact_plane_pose_in_cam_frame,
    outlier_prob,
    outlier_volume,
    model_box_dims,
):
    pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
        contact_param_sweep,
        face_param_sweep,
        model_box_dims[obj_idx],
        contact_plane_pose_in_cam_frame
    )

    # get best pose proposal
    rendered_object_images = renderer.render_parallel(pose_proposals, obj_idx)[...,:3]
    rendered_images = jax3dp3.splice_image_parallel(rendered_object_images, obs_point_cloud_image_complement)

    weights = jax3dp3.threedp3_likelihood_with_r_parallel_jit(
        obs_point_cloud_image, rendered_images, r_sweep, outlier_prob, outlier_volume
    )
    fully_occluded_weight = jax3dp3.threedp3_likelihood_jit(
        obs_point_cloud_image, obs_point_cloud_image, r_sweep[0], outlier_prob, outlier_volume
    )
    return pose_proposals, weights, fully_occluded_weight

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


def get_probabilites(hypotheses):
    scores = jnp.array( [i[0] for i in hypotheses])
    normalized_scores = jax3dp3.utils.normalize_log_scores(scores)
    return normalized_scores
