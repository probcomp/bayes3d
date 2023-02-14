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

MODEL_NAMES = ["002_master_chef_can"
,"003_cracker_box"
,"004_sugar_box"
,"005_tomato_soup_can"
,"006_mustard_bottle"
,"007_tuna_fish_can"
,"008_pudding_box"
,"009_gelatin_box"
,"010_potted_meat_can"
,"011_banana"
,"019_pitcher_base"
,"021_bleach_cleanser"
,"024_bowl"
,"025_mug"
,"035_power_drill"
,"036_wood_block"
,"037_scissors"
,"040_large_marker"
,"051_large_clamp"
,"052_extra_large_clamp"
,"061_foam_brick"]

def remove_zero_pad(img_id):
    for i, ch in enumerate(img_id):
        if ch != '0':
            return img_id[i:]


def get_test_img(scene_id, img_id, ycb_dir):
    if len(scene_id) < 6:
        scene_id = scene_id.rjust(6, '0')
    if len(img_id) < 6:
        img_id = img_id.rjust(6, '0')

    data_dir = os.path.join(ycb_dir, "test")
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
    cam_pose = jnp.linalg.inv(cam_pose_w2c)

    cam_depth_scale = image_cam_data['depth_scale']

    # get {visible mask, ID, pose} for each object in the scene
    anno = dict()

    # get GT object model ID+poses
    objects_gt_data = scene_imgs_gt_data[remove_zero_pad(img_id)]
    mask_visib_image_paths = sorted(glob(os.path.join(mask_visib_dir, f"{img_id}_*.png")))
    gt_ids = []
    anno = []
    gt_poses = []
    masks = []
    for object_gt_data, mask_visib_image_path in zip(objects_gt_data, mask_visib_image_paths):
        mask_visible = jnp.array(Image.open(mask_visib_image_path))

        model_R = jnp.array(object_gt_data['cam_R_m2c']).reshape(3,3)
        model_t = jnp.array(object_gt_data['cam_t_m2c']).reshape(3,1)
        model_pose = jnp.vstack([jnp.hstack([model_R, model_t]), jnp.array([0,0,0,1])])
        model_pose = model_pose.at[:3,3].set(model_pose[:3,3]*1.0/1000.0)
        gt_poses.append(model_pose)
        
        obj_id = object_gt_data['obj_id'] - 1

        gt_ids.append(obj_id)
        masks.append(jnp.array(mask_visible > 0))

    cam_pose = cam_pose.at[:3,3].set(cam_pose[:3,3]*1.0/1000.0)

    observation = jax3dp3.Jax3DP3Observation(
        rgb,
        depth * cam_depth_scale / 1000.0,
        cam_pose,
        rgb.shape[0],
        rgb.shape[1],
        cam_K[0,0],
        cam_K[1,1],
        cam_K[0,2],
        cam_K[1,2],
        0.01,
        10.0
    )

    return observation, gt_ids, jnp.array(gt_poses), masks


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
