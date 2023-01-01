import jax.numpy as jnp
import numpy as np
from .transforms_3d import add_homogenous_ones

def camera_rays_from_params(height, width, fx, fy, cx, cy):
    rows, cols = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    pixel_coords = jnp.stack([rows,cols],axis=-1)
    pixel_coords_dir = (pixel_coords - jnp.array([cx,cy])) / jnp.array([fx,fy])
    pixel_coords_dir_h = add_homogenous_ones(pixel_coords_dir)
    return pixel_coords_dir_h

def open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[0, 0] = fx
    persp[1, 1] = fy
    persp[0, 2] = cx
    persp[1, 2] = cy
    persp[2, 2] = near + far
    persp[2, 3] = near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:

    left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
    orth = np.array(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ]
    )
    return orth @ persp @ view

def scale_camera_paraeters(h,w,fx,fy,cx,cy, scaling_factor):
    new_fx = fx * scaling_factor
    new_fy = fy * scaling_factor
    new_cx = cx * scaling_factor
    new_cy = cy * scaling_factor

    new_h = int(np.round(h  * scaling_factor))
    new_w = int(np.round(w * scaling_factor))
    return new_h, new_w, new_fx, new_fy, new_cx, new_cy
