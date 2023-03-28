import jax3dp3 as j
import os
from tqdm import tqdm
import machine_common_sense as mcs
import numpy as np

def load_mcs_scene_data(scene_path):
    cache_dir = os.path.join(j.utils.get_assets_dir(), "mcs_cache")
    scene_name = scene_path.split("/")[-1]
    
    cache_filename = os.path.join(cache_dir, f"{scene_name}.npz")
    if os.path.exists(cache_filename):
        images = np.load(cache_filename,allow_pickle=True)["arr_0"]
    else:
        controller = mcs.create_controller(
            os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
        )

        scene_data = mcs.load_scene_json_file(scene_path)

        step_metadata = controller.start_scene(scene_data)
        image = j.RGBD.construct_from_step_metadata(step_metadata)

        step_metadatas = [step_metadata]
        for _ in tqdm(range(200)):
            step_metadata = controller.step("Pass")
            if step_metadata is None:
                break
            step_metadatas.append(step_metadata)

        all_images = []
        for i in tqdm(range(len(step_metadatas))):
            all_images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))

        images = all_images
        np.savez(cache_filename, images)

    return images

def update_pose_estimates(current_pose_estimates, image, renderer, all_enumerations, prior):
    intrinsics = renderer.intrinsics
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)

    for i in [*jnp.arange(pose_estimates.shape[0]), *jnp.arange(pose_estimates.shape[0])]:
        if i == 0 or i==1:
            continue

        pose_estimates_tiled = jnp.tile(
            pose_estimates[:,None,...], (1, all_enumerations.shape[0], 1, 1)
        )
        all_pose_proposals  = pose_estimates_tiled.at[i].set(
            jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], all_enumerations)
        )
        all_weights = batched_scorer_parallel_func(all_pose_proposals, point_cloud_image, renderer, num_batches=2)
        all_weights += prior_parallel(all_pose_proposals[i], pose_estimates[i], renderer.model_box_dims[i])
        
        pose_estimates = all_pose_proposals[:,all_weights.argmax(), :,:]

    return pose_estimates
