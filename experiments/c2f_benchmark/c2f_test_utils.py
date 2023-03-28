import jax3dp3.transforms_3d as t3d
from jax3dp3.renderer import RGBD
import jax3dp3 as j
import time
import numpy as np
import jax
import jax.numpy as jnp


R_SWEEP = jnp.array([0.02])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0

def run_grid(seg_id, sched, r_overlap_check=0.06, r_final = 0.07, viz=True):
    raise NotImplementedError("implement")

def run_c2f_contact_pose_only(renderer, image:RGBD, contact_plane_pose_in_cam_frame, scheds):
    raise NotImplementedError("TODO implement with tabletop inputs additionally")

def run_c2f_id_and_contact(renderer, image:RGBD, contact_plane_pose_in_cam_frame, scheds):
    intrinsics = renderer.intrinsics
    depth_scaled =  j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
    seg_scaled = j.utils.resize(image.segmentation_image, intrinsics.height, intrinsics.width)
    obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)
    
    all_segmentation_ids = np.unique(seg_scaled)
    all_segmentation_ids = all_segmentation_ids[all_segmentation_ids != -1]
    print(f"Given ground truth object ids: {all_segmentation_ids}")

    results = []
    for segmentation_id in all_segmentation_ids:
        depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, seg_scaled, segmentation_id, intrinsics)
        j.get_depth_image(depth_masked, max=intrinsics.far).save("masked.png")
        j.get_depth_image(depth_complement, max=intrinsics.far).save("complement.png")
        obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
        obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)

        hypotheses_over_time = j.c2f.c2f_full_pose(
            renderer,
            obs_point_cloud_image,
            obs_point_cloud_image_masked,
            obs_point_cloud_image_complement,
            scheds,
            contact_plane_pose_in_cam_frame,
            R_SWEEP,
            OUTLIER_PROB,
            OUTLIER_VOLUME,
            obj_idx_hypotheses=None
        )
        results.append(hypotheses_over_time)

    return results


def run_c2f_full_pose_only(renderer, image:RGBD, scheds):
    intrinsics = renderer.intrinsics
    depth_scaled =  j.utils.resize(image.depth, intrinsics.height, intrinsics.width)

    seg_scaled = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
    obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)

    all_segmentation_ids = np.unique(seg_scaled)
    all_segmentation_ids = all_segmentation_ids[all_segmentation_ids != 0]  # TODO what if the gt id is 0??
    print(f"Given ground truth object ids: {all_segmentation_ids}")

    results = []
    for segmentation_id in all_segmentation_ids:
        gt_idx = segmentation_id  # pose-only kubric dataset must have gt segmentation with gt obj id

        depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, seg_scaled, segmentation_id, intrinsics)
        j.get_depth_image(depth_masked, max=intrinsics.far).save("masked.png")
        j.get_depth_image(depth_complement, max=intrinsics.far).save("complement.png")
        obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
        obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)

        hypotheses_over_time = j.c2f.c2f_full_pose(
            renderer,
            obs_point_cloud_image,
            obs_point_cloud_image_masked,
            obs_point_cloud_image_complement,
            scheds,
            R_SWEEP,
            OUTLIER_PROB,
            OUTLIER_VOLUME,
            obj_idx_hypotheses=[gt_idx]
        )
        results.append(hypotheses_over_time)

    print("finished c2f")
    from IPython import embed; embed()
    return results


def run_c2f(renderer, image:RGBD, 
            scheds, 
            infer_id=True, 
            infer_contact=True, 
            viz=True):
    if infer_id and infer_contact:
        results = run_c2f_id_and_contact(renderer, image, scheds)
    if not infer_id and infer_contact:
        results = run_c2f_contact_pose_only(renderer, image, scheds)
    if not infer_id and not infer_contact:
        results = run_c2f_full_pose_only(renderer, image, scheds)
    if infer_id and not infer_contact:
        raise ValueError("TODO: Memory likely cannot support full pose inference + id inference")
    
    if not viz:
        return results, None
    else:
        results_over_time = results[0]
        score, gt_idx, best_pose = results_over_time[-1][0]
        rendered = renderer.render_single_object(best_pose, gt_idx)  # TODO ask nishad on cleanup
        rendered -= jnp.min(rendered)
        viz = j.viz.get_rgb_image(rendered, max=jnp.max(rendered))
        viz.save("best_pred.png")

        return results, viz  # TODO
    
