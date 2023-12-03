"""
Preprocess an unzipped .r3d file to the Record3DDataset format.
"""
import glob
import json
import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
import numpy as np
import png  # pip install pypng
import torch
import tyro
from natsort import natsorted
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm, trange
import subprocess

import bayes3d as b
import jax.numpy as jnp

def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_conf(filepath):
    with open(filepath, 'rb') as conf_fh:
        raw_bytes = conf_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    # depth_img = depth_img.reshape((640, 480))  # For a FaceID camera 3D Video
    conf_img = conf_img.reshape((256, 192))  # For a LiDAR 3D Video

    return conf_img


def load_color(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_color(outpath, img):
    cv2.imwrite(outpath, img)


def write_depth(outpath, depth):
    depth = depth * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(outpath)

def write_conf(outpath, conf):
    np.save(outpath, conf)

def write_pose(outpath, pose):
    np.save(outpath, pose.astype(np.float32))


def get_poses(metadata_dict: dict) -> int:
    """Converts Record3D's metadata dict into pose matrices needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
    Returns:
        np.array of pose matrices for each image of shape: (num_images, 4, 4)
    """

    poses_data = np.array(metadata_dict["poses"])  # (N, 3, 4)
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    camera_to_worlds = np.concatenate(
        [Rotation.from_quat(poses_data[:, :4]).as_matrix(), poses_data[:, 4:, None]],
        axis=-1,
    ).astype(np.float32)

    homogeneous_coord = np.zeros_like(camera_to_worlds[..., :1, :])
    homogeneous_coord[..., :, 3] = 1
    camera_to_worlds = np.concatenate([camera_to_worlds, homogeneous_coord], -2)
    return camera_to_worlds


def get_intrinsics(metadata_dict: dict):
    """Converts Record3D metadata dict into intrinsic info needed by nerfstudio
    Args:
        metadata_dict: Dict containing Record3D metadata
        downscale_factor: factor to scale RGB image by (usually scale factor is
            set to 7.5 for record3d 1.8 or higher -- this is the factor that downscales
            RGB images to lidar)
    Returns:
        dict with camera intrinsics keys needed by nerfstudio
    """

    # Camera intrinsics
    K = np.array(metadata_dict["K"]).reshape((3, 3)).T
    K = K
    fx = K[0, 0]
    fy = K[1, 1]

    H = metadata_dict["h"]
    W = metadata_dict["w"]

    # # TODO(akristoffersen): The metadata dict comes with principle points,
    # # but caused errors in image coord indexing. Should update once that is fixed.
    # cx, cy = W / 2, H / 2
    cx, cy = K[0, 2], K[1, 2]

    scaling_factor = metadata_dict["dw"] / metadata_dict["w"]

    intrinsics = b.Intrinsics(H, W, fx, fy, cx, cy, 0.01, 100.0)
    intrinsics_depth = b.scale_camera_parameters(intrinsics, scaling_factor)

    return intrinsics, intrinsics_depth

def load_r3d(r3d_path):
    r3d_path = Path(r3d_path)
    import subprocess
    subprocess.run([f"cp {r3d_path} /tmp/{r3d_path.name}.zip"], shell=True)
    subprocess.run([f"unzip -qq -o /tmp/{r3d_path.name}.zip -d /tmp/{r3d_path.name}"], shell=True)
    datapath = f"/tmp/{r3d_path.name}"

    f = open(os.path.join(datapath, "metadata"), "r")
    metadata = json.load(f)

    intrinsics, intrinsics_depth = get_intrinsics(metadata)

    color_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.jpg")))
    depth_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.depth")))
    conf_paths = natsorted(glob.glob(os.path.join(datapath, "rgbd", "*.conf")))

    colors = np.array([load_color(color_paths[i]) for i in range(len(color_paths))])
    depths = np.array([load_depth(depth_paths[i]) for i in range(len(color_paths))])
    depths[np.isnan(depths)] = 0.0

    poses = get_poses(metadata)
    P = np.array(  
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]
    )
    poses = (P @ poses @ P.T)

    return jnp.array(colors), jnp.array(depths), jnp.array(poses), intrinsics, intrinsics_depth