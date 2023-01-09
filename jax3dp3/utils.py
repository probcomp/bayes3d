import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax
import cv2
import jax3dp3.transforms_3d as t3d
import os
import pyransac3d
import sklearn.cluster

def get_assets_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),"assets")

def get_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")


def extract_2d_patches(data: jnp.ndarray, filter_shape: Tuple[int, int]) -> jnp.ndarray:
    """For each pixel, extract 2D patches centered at that pixel.
    Args:
        data (jnp.ndarray): Array of shape (H, W, ...)
            data needs to be 2, 3, or 4 dimensional.
        filter_shape (Tuple[int, int]): Size of the patches in H, W dimensions
    Returns:
        extracted_patches: Array of shape (H, W, filter_shape[0], filter_shape[1], C)
            extracted_patches[i, j] == data[
                i - filter_shape[0] // 2:i + filter_shape[0] - filter_shape[0] // 2,
                j - filter_shape[1] // 2:j + filter_shape[1] - filter_shape[1] // 2,
            ]
    """
    assert len(filter_shape) == 2
    output_shape = data.shape + filter_shape
    if data.ndim == 2:
        data = data[..., None, None]
    elif data.ndim == 3:
        data = data[:, :, None]

    padding = [
        (filter_shape[ii] // 2, filter_shape[ii] - filter_shape[ii] // 2 - 1)
        for ii in range(len(filter_shape))
    ]
    extracted_patches = jnp.moveaxis(
        jax.lax.conv_general_dilated_patches(
            lhs=data,
            filter_shape=filter_shape,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("HWNC", "OIHW", "HWNC"),
        ).reshape(output_shape),
        (-2, -1),
        (2, 3),
    )
    return extracted_patches


def make_cube_point_cloud(side_width, num_points):
    side_half_width = side_width / 2.0
    single_side = np.stack(np.meshgrid(
        np.linspace(-side_half_width, side_half_width, num_points),
        np.linspace(-side_half_width, side_half_width, num_points),
        np.linspace(0.0, 0.0, num_points)
    ),
        axis=-1
    ).reshape(-1,3)

    all_faces = []
    for a in [0,1,2]:
        for side in [-1.0, 1.0]:        
            perm = np.arange(3)
            perm[a] = 2
            perm[2] = a
            face = single_side[:,perm]
            face[:,a] = side * side_half_width
            all_faces.append(face)
    object_model_cloud = np.vstack(all_faces)
    return jnp.array(object_model_cloud)

def axis_aligned_bounding_box(object_points):
    maxs = jnp.max(object_points,axis=0)
    mins = jnp.min(object_points,axis=0)
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, t3d.transform_from_pos(center)

def find_plane(point_cloud, threshold):
    plane = pyransac3d.Plane()
    plane_eq, _ = plane.fit(point_cloud, threshold)
    plane_eq = np.array(plane_eq)
    plane_normal = np.array(plane_eq[:3])
    point_on_plane = plane_normal * -plane_eq[3]
    plane_x = np.cross(plane_normal, np.array([1.0, 0.0, 0.0]))
    plane_y = np.cross(plane_normal, plane_x)
    R = np.vstack([plane_x, plane_y, plane_normal]).T
    plane_pose = t3d.transform_from_rot_and_pos(R, point_on_plane)
    return plane_pose

def find_table_pose_and_dims(point_cloud, plane_pose, inlier_threshold, segmentation_threshold):
    points_in_plane_frame = t3d.apply_transform(point_cloud, jnp.linalg.inv(plane_pose))
    inliers = (jnp.abs(points_in_plane_frame[:,2]) < inlier_threshold)
    inlier_plane_points = points_in_plane_frame[inliers]
    inlier_table_points_seg = segment_point_cloud(inlier_plane_points, segmentation_threshold)
    
    table_points_in_plane_frame = inlier_plane_points[inlier_table_points_seg == 0]
    (cx,cy), (width,height), rotation_deg = cv2.minAreaRect(np.array(table_points_in_plane_frame[:,:2]))
    pose_shift = t3d.transform_from_rot_and_pos(
        t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(rotation_deg)),
        jnp.array([cx,cy, 0.0])
    )
    return plane_pose.dot(pose_shift), jnp.array([width, height, 0.000001])

def segment_point_cloud(point_cloud, threshold):
    c = sklearn.cluster.DBSCAN(eps=threshold).fit(point_cloud)
    labels = c.labels_
    return labels

def segment_point_cloud_image(point_cloud_image, threshold):
    point_cloud = point_cloud_image.reshape(-1,3)
    non_zero = point_cloud[:,2] > 0.0
    non_zero_indices = np.where(non_zero)[0]
    segmentation = segment_point_cloud(point_cloud[non_zero_indices,:], threshold)
    segmentation_img = np.ones(point_cloud.shape[0]) * -1.0 
    print(np.unique(segmentation))
    for (i,val) in enumerate(np.unique(segmentation)):
        segmentation_img[non_zero_indices[segmentation == val]] = i
    segmentation_img = segmentation_img.reshape(point_cloud_image.shape[:2])
    return segmentation_img