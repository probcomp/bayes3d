import jax3dp3.transforms_3d as t3d
from jax3dp3.renderer import RGBD
import jax3dp3 as j
import time
import numpy as np
import jax
import jax.numpy as jnp


R_SWEEP = jnp.array([0.03])
OUTLIER_PROB=0.1
OUTLIER_VOLUME = 1000.0**3
FILTER_SIZE=3

def run_grid(seg_id, sched, r_overlap_check=0.06, r_final = 0.07, viz=True):
    raise NotImplementedError("implement")

def run_c2f_id_only(renderer, image:RGBD, contact_plane_pose_in_cam_frame, scheds):
    raise NotImplementedError("TODO implement with tabletop inputs additionally")

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

def run_c2f_full_pose_only(renderer, image:RGBD, scheds, particles=1):
    """
    Run c2f for pose-only enumerative inference for object(s) with known ID/segmentation map
    """
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
        # j.get_depth_image(depth_masked, max=intrinsics.far).save("masked.png")
        # j.get_depth_image(depth_complement, max=intrinsics.far).save("complement.png")
        obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
        obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)

        hypotheses_over_time = j.c2f.c2f_full_pose(
            renderer,
            obs_point_cloud_image,
            obs_point_cloud_image_masked,
            obs_point_cloud_image_complement,
            scheds,
            r_sweep=R_SWEEP,
            outlier_prob=OUTLIER_PROB,
            outlier_volume=OUTLIER_VOLUME,
            filter_size=FILTER_SIZE,
            obj_idx_hypotheses=[gt_idx],
            top_k=particles
        )
        results.append(hypotheses_over_time)

    print("finished c2f")
    return results

def run_c2f(renderer, image:RGBD, 
            scheds, 
            infer_id=True, 
            infer_contact=True, 
            particles=1):
    if infer_id and infer_contact:
        results = run_c2f_id_and_contact(renderer, image, scheds, particles)
    if not infer_id and infer_contact:
        results = run_c2f_contact_pose_only(renderer, image, scheds, particles)
    if not infer_id and not infer_contact:
        results = run_c2f_full_pose_only(renderer, image, scheds, particles)
    if infer_id and not infer_contact:
        results = run_c2f_id_only(renderer, image, scheds, particles)
    
    return results
    
