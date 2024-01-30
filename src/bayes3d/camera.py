from collections import namedtuple

import jax.numpy as jnp
import numpy as np

import bayes3d as b

from .transforms_3d import add_homogenous_ones

# Declaring namedtuple()
Intrinsics = namedtuple(
    "Intrinsics", ["height", "width", "fx", "fy", "cx", "cy", "near", "far"]
)


def K_from_intrinsics(intrinsics):
    """Returns the camera matrix from the intrinsics.

    Args:
        intrinsics: (b.camera.Intrinsics) The intrinsics of the camera.
    Returns:
        (np.ndarray) The camera matrix K (3x3).
    """
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ]
    )


def scale_camera_parameters(intrinsics, scaling_factor):
    """Scale the camera parameters by a given factor.

    Args:
        intrinsics: (b.camera.Intrinsics) The intrinsics of the camera.
        scaling_factor: (float) The factor to scale the intrinsics by.
    Returns:
        (b.camera.Intrinsics) The scaled intrinsics.
    """
    new_fx = intrinsics.fx * scaling_factor
    new_fy = intrinsics.fy * scaling_factor
    new_cx = intrinsics.cx * scaling_factor
    new_cy = intrinsics.cy * scaling_factor

    new_h = int(np.round(intrinsics.height * scaling_factor))
    new_w = int(np.round(intrinsics.width * scaling_factor))
    return Intrinsics(
        new_h, new_w, new_fx, new_fy, new_cx, new_cy, intrinsics.near, intrinsics.far
    )


def camera_rays_from_intrinsics(intrinsics):
    """Returns the camera rays from the intrinsics.

    Args:
        intrinsics: (b.camera.Intrinsics) The intrinsics of the camera.
    Returns:
        (np.ndarray) The camera rays (height x width x 3).
    """
    rows, cols = jnp.meshgrid(
        jnp.arange(intrinsics.width), jnp.arange(intrinsics.height)
    )
    pixel_coords = jnp.stack([rows, cols], axis=-1)
    pixel_coords_dir = (
        pixel_coords - jnp.array([intrinsics.cx, intrinsics.cy])
    ) / jnp.array([intrinsics.fx, intrinsics.fy])
    pixel_coords_dir_h = add_homogenous_ones(pixel_coords_dir)
    return pixel_coords_dir_h


def project_cloud_to_pixels(point_cloud, intrinsics):
    """Project a point cloud to pixels.

    Args:
        point_cloud (jnp.ndarray): The point cloud. Shape (N, 3)
        intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
    Outputs:
        jnp.ndarray: The pixels. Shape (N, 2)
    """
    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * jnp.array([intrinsics.fx, intrinsics.fy])
    pixels = temp1 + jnp.array([intrinsics.cx, intrinsics.cy])
    return pixels


def render_point_cloud(point_cloud, intrinsics, pixel_smudge=1):
    """Render a point cloud to an image.

    Args:
        point_cloud (jnp.ndarray): The point cloud. Shape (N, 3)
        intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
    Outputs:
        jnp.ndarray: The image. Shape (height, width, 3)
    """
    transformed_cloud = point_cloud
    point_cloud = jnp.vstack([jnp.zeros((1, 3)), transformed_cloud])
    pixels = project_cloud_to_pixels(point_cloud, intrinsics)
    x, y = jnp.meshgrid(jnp.arange(intrinsics.width), jnp.arange(intrinsics.height))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (
        jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge
    )
    matches = matches * (intrinsics.far * 2.0 - point_cloud[:, -1][None, None, :])
    a = jnp.argmax(matches, axis=-1)
    return point_cloud[a]


def render_point_cloud_batched(point_cloud, intrinsics, NUM_PER, pixel_smudge=1):
    """Render a point cloud to an image in batches.

    Args:
        point_cloud (jnp.ndarray): The point cloud. Shape (N, 3)
        intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
    Outputs:
        jnp.ndarray: The image. Shape (height, width, 3)
    """
    all_images = []
    num_iters = jnp.ceil(point_cloud.shape[0] / NUM_PER).astype(jnp.int32)
    for i in range(num_iters):
        img = b.render_point_cloud(
            point_cloud[i * NUM_PER : i * NUM_PER + NUM_PER],
            intrinsics,
            pixel_smudge=pixel_smudge,
        )
        img = img.at[img[:, :, 2] < intrinsics.near].set(intrinsics.far)
        all_images.append(img)
    all_images_stack = jnp.stack(all_images, axis=-2)
    best = all_images_stack[:, :, :, 2].argmin(-1)
    img = all_images_stack[
        np.arange(intrinsics.height)[:, None],
        np.arange(intrinsics.width)[None, :],
        best,
        :,
    ]
    return img


def _open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = jnp.eye(4)
    view = view.at[1:3].set(view[1:3] * -1)

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp = jnp.array(
        [
            [fx, 0.0, -cx, 0.0],
            [0.0, -fy, -cy, 0.0],
            [0.0, 0.0, -near + far, near * far],
            [0.0, 0.0, -1, 0.0],
        ]
    )
    # persp[0, 0] = fx
    # persp[1, 1] = fy
    # persp[0, 2] = cx
    # persp[1, 2] = cy
    # persp[2, 2] = near + far
    # persp[2, 3] = near * far
    # persp[3, 2] = -1
    # # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    # persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:

    left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
    orth = jnp.array(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ]
    )
    return orth @ persp @ view


def getProjectionMatrix(intrinsics):
    top = intrinsics.near / intrinsics.fy * intrinsics.height / 2.0
    bottom = -top
    right = intrinsics.near / intrinsics.fy * intrinsics.height / 2.0
    left = -right
    P = jnp.zeros((4, 4))
    z_sign = 1.0
    P = P.at[0, 0].set(2.0 * intrinsics.near / (right - left))
    P = P.at[1, 1].set(2.0 * intrinsics.near / (top - bottom))
    P = P.at[0, 2].set((right + left) / (right - left))
    P = P.at[1, 2].set((top + bottom) / (top - bottom))
    P = P.at[2, 2].set(
        z_sign * (intrinsics.far + intrinsics.near) / (intrinsics.far - intrinsics.near)
    )
    P = P.at[2, 3].set(
        -2.0 * (intrinsics.far * intrinsics.near) / (intrinsics.far - intrinsics.near)
    )
    P = P.at[3, 2].set(z_sign)
    return jnp.transpose(P)
