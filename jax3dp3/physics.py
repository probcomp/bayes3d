import jax3dp3 as j
import os
from tqdm import tqdm
import machine_common_sense as mcs
import numpy as np
import jax
import jax.numpy as jnp

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
        while True:
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


def get_object_mask(point_cloud_image, segmentation, segmentation_ids):
    object_mask = jnp.zeros(point_cloud_image.shape[:2])
    object_ids = []
    for id in segmentation_ids:
        point_cloud_segment = point_cloud_image[segmentation == id]
        bbox_dims, pose = j.utils.aabb(point_cloud_segment)
        is_occluder = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
            (bbox_dims[0] < 0.1),
            (bbox_dims[1] < 0.1)),
            (bbox_dims[1] > 1.1)),
            (bbox_dims[0] > 1.1)),
            (bbox_dims[2] > 2.1)
        )
        if not is_occluder:
            object_mask += (segmentation == id)
            object_ids.append(id)

    object_mask = jnp.array(object_mask) > 0
    return object_ids, object_mask

def prior(new_state, prev_poses, prev_prev_poses, bbox_dims, known_id):    
    score = 0.0
    new_position = new_state[:3,3]
    bottom_of_object_y = new_position[1] + bbox_dims[known_id][1]/2.0

    prev_position = prev_poses[known_id][:3,3]
    prev_prev_position = prev_prev_poses[known_id][:3,3]

    velocity_prev = (prev_position - prev_prev_position) * jnp.array([1.0, 1.0, 0.25])
    velocity_with_gravity = velocity_prev + jnp.array([-jnp.sign(velocity_prev[0])*0.01, 0.1, 0.0])

    velocity_with_gravity2 = velocity_with_gravity * jnp.array([1.0 * (jnp.abs(velocity_with_gravity[0]) > 0.1), 1.0, 1.0 ])
    velocity = velocity_with_gravity2

    pred_new_position = prev_position + velocity

    score = score + jax.scipy.stats.multivariate_normal.logpdf(
        new_position, pred_new_position, jnp.diag(jnp.array([0.02, 0.02, 0.02]))
    )
    score += -100.0 * (bottom_of_object_y > 1.5)
    return score

prior_jit = jax.jit(prior)
prior_parallel_jit = jax.jit(jax.vmap(prior, in_axes=(0, None,  None, None, None)))
