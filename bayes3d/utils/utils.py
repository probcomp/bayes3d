import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax
import cv2
import bayes3d.transforms_3d as t3d
import bayes3d as b
import os
import pyransac3d
import sklearn.cluster
from jax.scipy.special import logsumexp
import time
import subprocess as sp
import os
import inspect

def make_onehot(n, i, hot=1, cold=0):
    return tuple(cold if j != i else hot for j in range(n))

def multivmap(f, args=None):
    if args is None:
        args = (True,) * len(inspect.signature(f).parameters)
    multivmapped = f
    for (i, ismapped) in reversed(list(enumerate(args))):
        if ismapped:
            multivmapped = jax.vmap(multivmapped, in_axes=make_onehot(len(args), i, hot=0, cold=None))
    return multivmapped


def time_code_block(func, args):
    start = time.time()
    output = func(*args)
    print(output[0])
    end = time.time()
    print ("Time elapsed:", end - start)

def get_assets_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"assets")

def make_cube_point_cloud(side_width, num_points):
    """
    Returns a point cloud of a cube.
    Args:
        side_width: (float) width of the cube
        num_points: (int) number of points per side
    Returns:
        object_model_cloud: (N,3) point cloud of the cube
    """
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

def discretize(data, resolution):
    """
        Discretizes a point cloud.
    """
    return jnp.round(data /resolution) * resolution


def voxelize(data, resolution):
    """
        Voxelize a point cloud.
    Args:
        data: (N,3) point cloud
        resolution: (float) resolution of the voxel grid
    Returns:
        data: (M,3) voxelized point cloud
    """
    data = discretize(data, resolution)
    data = jnp.unique(data, axis=0)
    return data

def aabb(object_points):
    """
    Returns the axis aligned bounding box of a point cloud.
    Args:
        object_points: (N,3) point cloud
    Returns:
        dims: (3,) dimensions of the bounding box
        pose: (4,4) pose of the bounding box
    """
    maxs = jnp.max(object_points,axis=0)
    mins = jnp.min(object_points,axis=0)
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, t3d.transform_from_pos(center)

def bounding_box_corners(dim):
    """
    Returns the corners of an axis aligned bounding box.
    Args:
        dim: (3,) dimensions of the bounding box
    Returns:
        corners: (8,3) corners of the bounding box    
    """
    corners = np.array([
        [-dim[0]/2, -dim[1]/2, -dim[2]/2],
        [dim[0]/2, -dim[1]/2, -dim[2]/2],
        [-dim[0]/2, dim[1]/2, -dim[2]/2],
        [dim[0]/2, dim[1]/2, -dim[2]/2],
        [-dim[0]/2, -dim[1]/2, dim[2]/2],
        [dim[0]/2, -dim[1]/2, dim[2]/2],
        [-dim[0]/2, dim[1]/2, dim[2]/2],
        [dim[0]/2, dim[1]/2, dim[2]/2]
    ])
    return corners

def plane_eq_to_plane_pose(plane_eq):
    """
    Returns the pose of a plane from its equation.
    Args:
        plane_eq: (4,) plane equation
    Returns:
        plane_pose: (4,4) pose of the plane
    """
    plane_eq = np.array(plane_eq)
    plane_normal = np.array(plane_eq[:3])
    point_on_plane = plane_normal * -plane_eq[3]
    plane_x = np.cross(plane_normal, np.array([1.0, 0.0, 0.0]))
    plane_y = np.cross(plane_normal, plane_x)
    R = np.vstack([plane_x, plane_y, plane_normal]).T
    plane_pose = b.t3d.transform_from_rot_and_pos(R, point_on_plane)
    return plane_pose

def find_plane(point_cloud, threshold,  minPoints=100, maxIteration=1000):
    """
    Returns the pose of a plane from a point cloud.
    """
    plane = pyransac3d.Plane()
    plane_eq, _ = plane.fit(point_cloud, threshold, minPoints=minPoints, maxIteration=maxIteration)
    plane_pose = plane_eq_to_plane_pose(plane_eq)
    return plane_pose

def get_bounding_box_z_axis_aligned(point_cloud):
    """
    Returns the axis aligned bounding box of a point cloud.
    """
    dims, pose = aabb(point_cloud)
    point_cloud_centered = t3d.apply_transform(point_cloud, t3d.inverse_pose(pose))
    
    (cx,cy), (width,height), rotation_deg = cv2.minAreaRect(np.array(point_cloud_centered[:,:2]))
    pose_shift = t3d.transform_from_rot_and_pos(
        t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(rotation_deg)),
        jnp.array([cx,cy, 0.0])
    )
    new_pose = pose @ pose_shift
    dims, _ = aabb( t3d.apply_transform(point_cloud, t3d.inverse_pose(new_pose)))
    return dims, new_pose

def find_plane_and_dims(point_cloud, ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.008):
    """
    Returns the pose of a plane from a point cloud.
    Args:
        point_cloud: (N,3) point cloud
        ransac_threshold: (float) ransac threshold for plane fitting
    Returns:
        plane_pose: (4,4) pose of the plane
    """
    plane_pose = find_plane(np.array(point_cloud), ransac_threshold)
    points_in_plane_frame = t3d.apply_transform(point_cloud, jnp.linalg.inv(plane_pose))
    inliers = (jnp.abs(points_in_plane_frame[:,2]) < inlier_threshold)
    inlier_plane_points = points_in_plane_frame[inliers]
    inlier_table_points_seg = segment_point_cloud(inlier_plane_points, segmentation_threshold)

    most_frequent_seg_id = get_largest_cluster_id_from_segmentation(inlier_table_points_seg)
    
    table_points_in_plane_frame = inlier_plane_points[inlier_table_points_seg == most_frequent_seg_id]

    (cx,cy), (width,height), rotation_deg = cv2.minAreaRect(np.array(table_points_in_plane_frame[:,:2]))
    pose_shift = t3d.transform_from_rot_and_pos(
        t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(rotation_deg)),
        jnp.array([cx,cy, 0.0])
    )
    table_pose = plane_pose.dot(pose_shift)
    table_dims = jnp.array([width, height, 1e-10])
    return table_pose, table_dims

def segment_point_cloud(point_cloud, threshold=0.01, min_points_in_cluster=0):
    c = sklearn.cluster.DBSCAN(eps=threshold).fit(point_cloud)
    labels = c.labels_
    unique, counts =  np.unique(labels, return_counts=True)
    for val in unique[counts < min_points_in_cluster]:
        labels[labels == val] = -1
    return labels

def segment_point_cloud_image(point_cloud_image, threshold=0.01, min_points_in_cluster=0):
    point_cloud = point_cloud_image.reshape(-1,3)
    non_zero = point_cloud[:,2] > 0.0
    segmentation_img = np.ones(point_cloud.shape[0]) * -1.0
    if non_zero.sum() == 0:
        return segmentation_img
    non_zero_indices = np.where(non_zero)[0]
    segmentation = segment_point_cloud(point_cloud[non_zero_indices,:], threshold=threshold, min_points_in_cluster=min_points_in_cluster)
    unique, counts =  np.unique(segmentation, return_counts=True)
    for (i,val) in enumerate(unique[unique != -1]):
        segmentation_img[non_zero_indices[segmentation == val]] = i
    segmentation_img = segmentation_img.reshape(point_cloud_image.shape[:2])
    return segmentation_img

def get_largest_cluster_id_from_segmentation(segmentation_array_or_img):
    """
    Returns the id of the largest cluster in a segmentation.
    """
    unique, counts =  jnp.unique(segmentation_array_or_img, return_counts=True)
    non_neg_one = (unique != -1)
    unique = unique[non_neg_one]
    counts = counts[non_neg_one]
    return unique[counts.argmax()]

def normalize_log_scores(log_p):
    """
    Normalizes log scores.
    Args:
        log_p: (N,) log scores
    Returns:
        log_p_normalized: (N,) normalized log scores
    """
    return jnp.exp(log_p - logsumexp(log_p))

def resize(depth, h, w):
    """
    Resizes a depth image.
    Args:
        depth: (H,W) depth image
        h: (int) new height
        w: (int) new width
    Returns:
        depth: (h,w) resized depth image
    """
    return cv2.resize(np.asarray(depth, dtype=depth.dtype), (w,h),interpolation=0).astype(depth.dtype)

def scale(depth, factor):
    """
    Scales a depth image.
    Args:
        depth: (H,W) depth image
    Returns:
        depth: (h,w) scaled depth image
    """
    h,w = depth.shape[:2]
    return resize(depth, int(h * factor), int(w * factor))

def infer_table_plane(point_cloud_image, camera_pose, intrinsics, ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.008):
    point_cloud_flat = point_cloud_image.reshape(-1, 3)
    point_cloud_flat_not_far = point_cloud_flat[point_cloud_flat[:,2] < intrinsics.far, :]
    table_pose, table_dims = find_plane_and_dims(
        t3d.apply_transform(point_cloud_flat_not_far, camera_pose), 
        ransac_threshold=ransac_threshold, inlier_threshold=inlier_threshold, segmentation_threshold=segmentation_threshold
    )

    table_pose_in_cam_frame = t3d.inverse_pose(camera_pose) @ table_pose
    if table_pose_in_cam_frame[2,2] > 0:
        table_pose = table_pose @ t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
    return table_pose, table_dims

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def make_schedules_contact_params(grid_widths, rotation_angle_widths, grid_params):
    sched = []

    for (grid_width, angle_width, grid_param) in zip(grid_widths, rotation_angle_widths, grid_params):
        cf = b.scene_graph.enumerate_contact_and_face_parameters(
            -grid_width, -grid_width, -angle_width, 
            +grid_width, +grid_width, angle_width, 
            *grid_param,  # *grid_param is num_x, num_y, num_angle
            jnp.arange(6)
        )
        sched.append(cf)
    return sched

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
