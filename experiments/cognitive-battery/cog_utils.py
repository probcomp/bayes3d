import numpy as np
import jax.numpy as jnp
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

import json


def get_object_transforms(obj_name, init_transform):
    with open("object_transforms.json") as f:
        object_transforms = json.load(f)

    if obj_name not in object_transforms:
        return [init_transform]

    transforms = []
    for transform in object_transforms[obj_name]:
        new_transform = init_transform.copy()
        object_transforms[obj_name]
        if "rot" in transform:
            rot = R.from_euler("zyx", transform["rot"], degrees=True).as_matrix()
            new_transform = new_transform.at[:3, :3].set(jnp.array(rot))
        if "scale" in transform:
            scale = jnp.array(transform["scale"])
            new_transform = new_transform.at[:3, :3].set(new_transform[:3, :3] * scale)
        if "pos_delta" in transform:
            t = jnp.array(transform["pos_delta"]).reshape(3, 1)
            delta_pose = jnp.vstack(
                [jnp.hstack([jnp.eye(3), t]), jnp.array([0.0, 0.0, 0.0, 1.0])]
            )
            new_transform = new_transform.dot(delta_pose)
        transforms.append(new_transform)
    return transforms


def get_camera_intrinsics(width, height, fov):
    cx, cy = width / 2.0, height / 2.0
    aspect_ratio = width / height
    fov_y = np.deg2rad(fov)
    fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
    fx = cx / np.tan(fov_x / 2.0)
    fy = cy / np.tan(fov_y / 2.0)

    return fx, fy, cx, cy


def cluster_camera_coords(coords, eps=0.1, min_points_per_shape=5):
    # Given an array of depth coordinates [w, h, 3], finds clusters of points that belong to the same shape

    flat_coords = coords.reshape(-1, 3)
    clustering = DBSCAN(eps=eps, min_samples=min_points_per_shape, n_jobs=-1).fit(
        flat_coords
    )
    return clustering
