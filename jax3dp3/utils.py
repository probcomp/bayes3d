import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax

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

def make_centered_grid_enumeration_3d_points(x,y,z,num_x,num_y,num_z):
    gridding = jnp.linspace(-1.0, 1.0, 5)
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(-x,x,num_x),
        jnp.linspace(-y,y,num_y),
        jnp.linspace(-z,z,num_z)
    ),
        axis=-1)
    deltas = deltas.reshape(-1,3)
    return deltas

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


def depth_to_coords_in_camera(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to coords in camera space for points in mask.
    Args:
        depth: Array of shape (H, W).
        intrinsics: Array of shape (3, 3), camera intrinsic matrix.
        mask: Array of shape (H, W), with 1s where points are quried.
        as_image_shape: If True, return arrays of shape (H, W, 3)
    Returns:
        np.ndarray: Array of shape (N, 3) or (H, W, 3), coordinates in camera space.
        np.ndarray: Array of shape (N, 2) or (H, W, 2), coordinates on image plane.
            N is the number of 1s in mask.
    """
    vu = np.mgrid[: depth.shape[0], : depth.shape[1]]

    depth_for_uv = depth[vu[0], vu[1]]
    full_vec = np.stack(
        [vu[1] * depth_for_uv, vu[0] * depth_for_uv, depth_for_uv], axis=0
    )
    coords_in_camera = np.moveaxis(
        np.einsum('ij,j...->i...', np.linalg.inv(intrinsics), full_vec), 0, -1
    )
    coords_on_image = np.moveaxis(vu, 0, -1)
    return coords_in_camera, coords_on_image


