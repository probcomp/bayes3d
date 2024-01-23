import kubric as kb
import numpy as np


def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.0,
    outer_radius: float = 12.0,
    z_offset: float = 0.1,
):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        camera_start = np.array(
            kb.sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset)
        )
        direction = rng.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement
        if (
            inner_radius <= np.linalg.norm(camera_end) <= outer_radius
            and camera_end[2] > z_offset
        ):
            return camera_start, camera_end
