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

