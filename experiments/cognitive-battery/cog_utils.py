import json
import os
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

import bayes3d
from bayes3d.renderer import _Renderer
from bayes3d.viz import get_depth_image

def get_object_transforms(obj_name: str, init_transform: jnp.ndarray) -> jnp.ndarray:
    """
    Recovers the universal transforms for each mesh from an annotated file of transforms.
    Transforms refer to positional translations, rotations, and scale factors used in the
    renderer.
    """

    with open("resources/object_transforms.json") as f:
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


def get_camera_intrinsics(
    width: float, height: float, fov: float, near: float = 0.001, far: float = 50.0
) -> bayes3d.Intrinsics:
    """
    Recovers the camera intrinsics details given image and capture parameters.
    """

    cx, cy = width / 2.0, height / 2.0
    aspect_ratio = width / height
    fov_y = np.deg2rad(fov)
    fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
    fx = cx / np.tan(fov_x / 2.0)
    fy = cy / np.tan(fov_y / 2.0)
    return bayes3d.Intrinsics(height, width, fx, fy, cx, cy, near, far)


def find_best_mesh(
    renderer: _Renderer,
    meshes: List[str],
    obj_transform: jnp.ndarray,
    depth: jnp.ndarray,
) -> Union[None, Tuple[int, jnp.ndarray]]:
    """
    Finds the mesh that best explains a point cloud given a list of the available mesh names.
    """

    best = None
    k = np.inf
    for m in range(len(meshes)):
        obj_transforms = get_object_transforms(meshes[m], obj_transform)
        for i, transform in enumerate(obj_transforms):
            rendered_image = renderer.render_single_object(transform, m)
            get_depth_image(rendered_image[:, :, 2], max=50).save(pp + f"{meshes[m]}_{i}.png")
            scaling_factor = (rendered_image[:, :, 2] != rendered_image.max()).sum() / (
                depth[:, :, 2] != 0.0
            ).sum()
            scaling_factor = (
                1 / scaling_factor if scaling_factor < 1 else scaling_factor
            )
            keep_points = (
                jnp.sum(
                    jnp.logical_or(
                        ((depth[:, :, 2] != 0.0) * (rendered_image[:, :, 2] == rendered_image.max())),
                        ((depth[:, :, 2] == 0.0) * (rendered_image[:, :, 2] != rendered_image.max())),
                    )
                )
                * scaling_factor
            )
            if keep_points < k:
                k = keep_points
                best = (m, transform)
    return best


def check_occlusion(
    renderer: _Renderer,
    pose_estimates: jnp.ndarray,
    indices: List[int],
    obj_idx: int,
    occlusion_threshold: int = 10,
) -> bool:
    """
    Checks whether an object is occluded by comparing the depth map rendered with
    and without the object respectively. If there is more than occlusion_threshold
    unexplained points, then the object is assumed to be non-occluded.
    """

    depth_wi = renderer.render_multiobject(pose_estimates, indices)
    no_obj_mask = jnp.where(jnp.arange(pose_estimates.shape[0]) != obj_idx)
    depth_woi = renderer.render_multiobject(
        pose_estimates[no_obj_mask], jnp.array(indices)[no_obj_mask]
    )
    return jnp.sum(depth_wi[:, :, 2] != depth_woi[:, :, 2]) < occlusion_threshold


def check_containment(
    renderer: _Renderer,
    pose_estimates: jnp.ndarray,
    indices: List[int],
    obj_idx: int,
    containment_threshold: float = 0.03,
) -> Union[None, int]:
    """
    Checks whether an object is contained by any other object, and if so returns the index
    of the object containing it. Containment occurs when the bounding box of the object of
    interest lies almost entirely within the bounding box of any other object, within some
    threshold for error `containment_threshold`. Returns `None` if no containment is detected,
    else returns the index of the containing object.
    """

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

    contained = jnp.all(closest_maxs > (i_maxs - containment_threshold)) and jnp.all(
        closest_mins < (i_mins + containment_threshold)
    )

    return closest_idx.item() if contained else None


def full_translation_deltas_single(
    poses: jnp.ndarray, translation_deltas: jnp.ndarray, index: int
) -> jnp.ndarray:
    """
    Creates a list of multiobject translation proposals by applying translation deltas to the single
    object at index `index` and keeping the poses for all other objects the same.
    """

    translation_deltas_full = jnp.tile(
        jnp.eye(4)[None, :, :],
        (translation_deltas.shape[0], poses.shape[0], 1, 1),
    )
    translation_deltas_full = translation_deltas_full.at[:, index, :, :].set(
        translation_deltas
    )
    translation_proposals = jnp.einsum("bij,abjk->abik", poses, translation_deltas_full)
    return translation_proposals


def get_reward_idx(
    meshes: List[str], indices: List[int], reward_mesh_name: str = "apple"
) -> int:
    assert (reward_mesh_name in meshes), f"Could not find mesh name {reward_mesh_name} in the meshes list."
    reward_mesh_idx = meshes.index(reward_mesh_name)

    assert reward_mesh_idx in indices, "Could not locate the reward in the scene."
    return indices.index(reward_mesh_idx)


def read_label(video_path: str, label_key: str = "final_location") -> int:
    expt_stats_path = os.path.join(video_path, "human_readable/experiment_stats.yaml")
    with open(expt_stats_path) as f:
        expt_stats = yaml.safe_load(f)
        label = expt_stats[label_key]
    return label
