import sys
import os
sys.path.append(os.getcwd())

from dataclasses import dataclass
from glob import glob
from PIL import Image

import json
import jax3dp3
import jax.numpy as jnp
import numpy as np
import pickle


def remove_zero_pad(img_id):
    for i, ch in enumerate(img_id):
        if ch != '0':
            return img_id[i:]


def get_test_img(scene_id, img_id, data_dir):
    if len(scene_id) < 6:
        scene_id = scene_id.rjust(6, '0')
    if len(img_id) < 6:
        img_id = img_id.rjust(6, '0')

    scene_data_dir = os.path.join(data_dir, scene_id)  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json

    scene_rgb_images_dir = os.path.join(scene_data_dir, 'rgb')
    scene_depth_images_dir = os.path.join(scene_data_dir, 'depth')
    mask_visib_dir = os.path.join(scene_data_dir, 'mask_visib')

    with open(os.path.join(scene_data_dir, "scene_camera.json")) as scene_cam_data_json:
        scene_cam_data = json.load(scene_cam_data_json)

    with open(os.path.join(scene_data_dir, "scene_gt.json")) as scene_imgs_gt_data_json:
        scene_imgs_gt_data = json.load(scene_imgs_gt_data_json)
    
    # get rgb image
    rgb = jnp.array(Image.open(os.path.join(scene_rgb_images_dir, f"{img_id}.png")))

    # get depth image
    depth = jnp.array(Image.open(os.path.join(scene_depth_images_dir, f"{img_id}.png")))
        
    # get camera intrinsics and pose for image
    image_cam_data = scene_cam_data[remove_zero_pad(img_id)]

    cam_K = jnp.array(image_cam_data['cam_K']).reshape(3,3)
    cam_R_w2c = jnp.array(image_cam_data['cam_R_w2c']).reshape(3,3)
    cam_t_w2c = jnp.array(image_cam_data['cam_t_w2c']).reshape(3,1)
    cam_pose_w2c = jnp.vstack([jnp.hstack([cam_R_w2c, cam_t_w2c]), jnp.array([0,0,0,1])])
    cam_depth_scale = image_cam_data['depth_scale']

    # get {visible mask, ID, pose} for each object in the scene
    anno = dict()

    # get GT object model ID+poses
    objects_gt_data = scene_imgs_gt_data[remove_zero_pad(img_id)]
    mask_visib_image_paths = sorted(glob(os.path.join(mask_visib_dir, f"{img_id}_*.png")))
    gt_ids = []
    anno = []
    for object_gt_data, mask_visib_image_path in zip(objects_gt_data, mask_visib_image_paths):
        mask_visible = jnp.array(Image.open(mask_visib_image_path))

        cam_R_m2c = jnp.array(object_gt_data['cam_R_m2c']).reshape(3,3)
        cam_t_m2c = jnp.array(object_gt_data['cam_t_m2c']).reshape(3,1)
        cam_pose_m2c = jnp.vstack([jnp.hstack([cam_R_m2c, cam_t_m2c]), jnp.array([0,0,0,1])])
        
        obj_id = object_gt_data['obj_id']

        gt_ids.append(obj_id)
        anno.append({'mask_visible': mask_visible, 'gt_poses_m2c': cam_pose_m2c})


    # Create the TestImage instance for the image
    ycbv_img = BOPTestImage(dataset=data_dir,
                            scene_id=scene_id, 
                            img_id=img_id,
                            rgb=rgb,
                            depth=depth,
                            intrinsics=cam_K,
                            camera_pose=cam_pose_w2c,
                            bop_obj_indices=gt_ids,
                            annotations=anno  # mask and gt poses
                            )
    print(f"Retrieved test scene {scene_id} image {img_id} with {len(objects_gt_data)} objects")
    return ycbv_img


def create_ycbv_dataset(data_dirname: str) -> dict:
    YCBV_DATA = {}
    data_dir = os.path.join(jax3dp3.utils.get_data_dir(), data_dirname)

    scene_ids = os.listdir(data_dir)

    for scene_id in scene_ids:
        YCBV_DATA[scene_id] = []
        scene_data_dir = os.path.join(data_dir, scene_id)  # depth, mask, mask_visib, rgb; scene_camera.json, scene_gt_info.json, scene_gt.json
        scene_rgb_images_dir = os.path.join(scene_data_dir, 'rgb')

        for img_filename in os.listdir(scene_rgb_images_dir):
            img_id = img_filename.split(".")[0]

            ycbv_img = get_test_img(scene_id, img_id, data_dirname)

            YCBV_DATA[scene_id].append(ycbv_img)
            print(f"Processed scene {scene_id} image {img_id}")
        
        print('\n')
        break  # to do just one scene
        
    return YCBV_DATA

@dataclass
class BOPTestImage:
    """BOPTestImage.
    Args:
        rgb: Array of shape (H, W, 3)
        depth: Array of shape (H, W). In meters
        intrinsics: Array of shape (3, 3)
        bop_obj_indices: BOP indices of the objects in the image.
            Ranges from 1 to 21.
        annotations:
            model_to_cam: Array of shape (4, 4)
            mask: Array of shape (H, W)
            mask_visible: Array of shape (H, W)
    """

    dataset: str
    scene_id: int
    img_id: int
    rgb: jnp.ndarray
    depth: jnp.ndarray
    intrinsics: jnp.ndarray
    camera_pose: jnp.ndarray
    bop_obj_indices: list
    annotations: list
    # default_scales: np.ndarray

    def get_scene_img_ids(self):
        return (self.scene_id, self.img_id)

    def get_camera_intrinsics(self):
        cam_K = self.intrinsics
        fx, fy, cx, cy = cam_K[0][0], cam_K[1][1], cam_K[0][2], cam_K[1][2]
        return fx, fy, cx, cy
    
    def get_image_dims(self) -> tuple:
        return self.depth.shape
    
    def get_camera_pose(self):
        return self.camera_pose

    def get_rgb_image(self):
        return self.rgb
    
    def get_depth_image(self):
        return self.depth 

    def get_gt_indices(self):
        return self.bop_obj_indices

    def get_object_masks(self):
        obj_masks = [anno['mask_visible'] for anno in self.annotations]
        return obj_masks

    def get_gt_poses(self):
        gt_poses = [anno['gt_poses_m2c'] for anno in self.annotations]
        return gt_poses





# dataset = create_ycbv_dataset('ycbv_test')
# scene_data = dataset['000054']

# img_id = 0
# image = scene_data[img_id]

# K = image.get_camera_intrinsics()
# gt_indices = image.get_gt_indices()
# gt_poses = image.get_gt_poses()
# cam_pose = image.get_camera_pose()


# get_test_img('000054', '001568')




# from IPython import embed; embed()
