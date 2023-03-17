import numpy as np
import jax.numpy as jnp
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

import json
from jax3dp3 import Renderer


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


def check_occlusion(renderer: Renderer, pose_estimates, indices, obj_idx, occlusion_threshold=10):
    depth_wi = renderer.render_multiobject(pose_estimates, indices)
    no_obj_mask = jnp.where(jnp.arange(pose_estimates.shape[0]) != obj_idx)
    depth_woi = renderer.render_multiobject(
        pose_estimates[no_obj_mask], jnp.array(indices)[no_obj_mask]
    )
    return jnp.sum(depth_wi[:, :, 2] != depth_woi[:, :, 2]) < occlusion_threshold


def check_containment(renderer: Renderer, pose_estimates, indices, obj_idx, containment_threshold=0.03):
    positions = pose_estimates[:, :-1, -1]
    diff = jnp.square(positions - positions[obj_idx]).sum(-1)
    closest_idx = jnp.argsort(diff)[1]

    depth_closest = renderer.render_single_object(
        pose_estimates[closest_idx], indices[closest_idx]
    )
    closest_mask = depth_closest[:, :, 2] != 0
    closest_points = depth_closest[closest_mask]
    closest_maxs = np.max(closest_points, axis=0)
    closest_mins = np.min(closest_points, axis=0)

    depth_i = renderer.render_single_object(pose_estimates[obj_idx], indices[obj_idx])
    i_mask = depth_i[:, :, 2] != 0
    i_points = depth_i[i_mask]
    if not i_points.size:
        return closest_idx.item()
    i_maxs = np.max(i_points, axis=0)
    i_mins = np.min(i_points, axis=0)
    
    contained = jnp.all(closest_maxs > (i_maxs - containment_threshold)) \
        and jnp.all(closest_mins < (i_mins + containment_threshold))
    
    return closest_idx.item() if contained else None
        
def find_best_mesh(renderer: Renderer, meshes, obj_transform, depth):
    best = None
    k = np.inf
    for m in range(len(meshes)):
        obj_transforms = get_object_transforms(meshes[m], obj_transform)
        for i, transform in enumerate(obj_transforms):
            rendered_image = renderer.render_single_object(transform, m)
            keep_points = (
                jnp.sum(
                    jnp.logical_or(
                        (
                            (depth[:, :, 2] != 0.0)
                            * (rendered_image[:, :, 2] == 0)
                        ),
                        (
                            (depth[:, :, 2] == 0.0)
                            * (rendered_image[:, :, 2] != 0)
                        ),
                    )
                )
                / (rendered_image[:, :, 2] != 0.0).sum()
            )
            if keep_points < k:
                k = keep_points
                best = (m, transform)
    return best
