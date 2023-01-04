import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax
import jax3dp3.transforms_3d as t3d
import os

def get_assets_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),"assets")

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