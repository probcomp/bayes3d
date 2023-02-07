import collections
import heapq
import jax
import jax.numpy as jnp
import jax3dp3
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
    return contact_param_sched, face_param_sched


def c2f_contact_parameters(
    hypotheses,
    contact_param_sched,
    face_param_sched,
    r,
    contact_plane_pose,
    obs_point_cloud_image,
    obs_image_complement,
    outlier_prob,
    outlier_volume,
    model_box_dims
):
    hypotheses_over_time = [hypotheses]

    for c2f_iter in range(len(contact_param_sched)):
        contact_param_sweep_delta, face_param_sweep = contact_param_sched[c2f_iter], face_param_sched[c2f_iter]
        new_hypotheses = []

        for hyp in hypotheses:
            old_score = hyp[0]
            obj_idx = hyp[1]
            contact_params = hyp[2]

            new_contact_param_sweep = contact_params + contact_param_sweep_delta
            
            pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
                new_contact_param_sweep,
                face_param_sweep,
                model_box_dims[obj_idx],
                contact_plane_pose
            )

            # get best pose proposal
            rendered_object_images = jax3dp3.render_parallel(pose_proposals, obj_idx)[...,:3]
            rendered_images = jax3dp3.splice_in_object_parallel(rendered_object_images, obs_image_complement)

            weights = jax3dp3.threedp3_likelihood_parallel_jit(
                obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume
            )
            best_idx = jnp.argmax(weights)

            new_hypotheses.append(
                (
                    weights[best_idx],
                    obj_idx,
                    new_contact_param_sweep[best_idx],
                    pose_proposals[best_idx]
                )
            )

        hypotheses_over_time.append(new_hypotheses)
        hypotheses = new_hypotheses
    return hypotheses_over_time

def get_probabilites(hypotheses):
    scores = jnp.array( [i[0] for i in hypotheses])
    normalized_scores = jax3dp3.utils.normalize_log_scores(scores)
    return normalized_scores

def c2f_viz(rgb, hypotheses_over_time, names, camera_params, probabilities=None):
    if probabilities is None:
        probabilities = get_probabilites(hypotheses_over_time[-1])

    (h,w,fx,fy,cx,cy, near, far) = camera_params
    orig_h, orig_w = rgb.shape[:2]
    num_objects = len(hypotheses_over_time[0])
    rgb_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb, 255.0), orig_h, orig_w)

    viz_panels = []
    for idx in range(num_objects):
        viz_images = []
        for hypotheses in hypotheses_over_time:
            (score, obj_idx, _, pose) = hypotheses[idx]
            depth = jax3dp3.render_single_object(pose, obj_idx)
            depth_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(depth[:,:,2], max=far), orig_h, orig_w)
            viz_images.append(
                jax3dp3.viz.overlay_image(
                    rgb_viz,
                    depth_viz
                )
            )
        viz_panels.append(
            jax3dp3.viz.multi_panel(
                viz_images, title="{:s}    -   Probability: {:0.3f}".format(names[idx], probabilities[idx])
            )
        )
    return viz_panels


###### Occlusion check
def c2f_occluded_object_pose_distribution(
    obj_idx,
    segmentation_image,
    contact_param_sweep,
    face_param_sweep,
    r,
    contact_plane_pose,
    obs_point_cloud_image,
    outlier_prob,
    outlier_volume,
    model_box_dims,
    camera_params,
    threshold
):
    pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
        contact_param_sweep,
        face_param_sweep,
        model_box_dims[obj_idx],
        contact_plane_pose
    )


    obj_idx = 3
    rendered_object_images = jax3dp3.render_parallel(pose_proposals, obj_idx)[...,:3]
    rendered_images = jax3dp3.splice_in_object_parallel(rendered_object_images, obs_point_cloud_image)

    weights = jax3dp3.threedp3_likelihood_parallel_jit(
        obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume
    )
    
    good_poses = pose_proposals[weights >= weights.max() - 0.1]

    good_pose_centers = good_poses[:,:3,-1]
    good_pose_centers_cam_frame = good_pose_centers

    (h,w,fx,fy,cx,cy, near, far) = camera_params
    pixels = jax3dp3.project_cloud_to_pixels(good_pose_centers_cam_frame,fx,fy,cx,cy).astype(jnp.int32)

    seg_ids = []
    for (x,y) in pixels:
        if 0<= x < w and 0 <= y < h:
            seg_ids.append(segmentation_image[int(y),int(x)])
    ranked_high_value_seg_ids =jnp.unique(jnp.array(seg_ids))
    ranked_high_value_seg_ids = ranked_high_value_seg_ids[ranked_high_value_seg_ids != -1]

    return good_poses, pose_proposals, ranked_high_value_seg_ids


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
        [rgb_viz, enumeration_viz_all, overlay_viz, best_occluder],
        labels=["RGB", "Enumeration", "Pose Dist.", "Occluder"],
    )