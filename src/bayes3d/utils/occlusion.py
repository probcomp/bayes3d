import jax
import jax.numpy as jnp

import bayes3d as b


def voxel_occupied_occluded_free(camera_pose, depth_image, grid, intrinsics, tolerance):
    grid_in_cam_frame = b.apply_transform(grid, b.t3d.inverse_pose(camera_pose))
    pixels = b.project_cloud_to_pixels(grid_in_cam_frame, intrinsics).astype(jnp.int32)
    valid_pixels = (
        (0 <= pixels[:, 0])
        * (0 <= pixels[:, 1])
        * (pixels[:, 0] < intrinsics.width)
        * (pixels[:, 1] < intrinsics.height)
    )
    real_depth_vals = depth_image[pixels[:, 1], pixels[:, 0]] * valid_pixels + (
        1 - valid_pixels
    ) * (intrinsics.far + 1.0)

    projected_depth_vals = grid_in_cam_frame[:, 2]
    occupied = jnp.abs(real_depth_vals - projected_depth_vals) < tolerance
    occluded = real_depth_vals < projected_depth_vals
    occluded = occluded * (1.0 - occupied)
    _free = (1.0 - occluded) * (1.0 - occupied)
    return 1.0 * occupied + 0.5 * occluded


voxel_occupied_occluded_free_jit = jax.jit(voxel_occupied_occluded_free)
voxel_occupied_occluded_free_parallel_camera = jax.jit(
    jax.vmap(voxel_occupied_occluded_free, in_axes=(0, None, None, None, None))
)
voxel_occupied_occluded_free_parallel_camera_depth = jax.jit(
    jax.vmap(voxel_occupied_occluded_free, in_axes=(0, 0, None, None, None))
)
