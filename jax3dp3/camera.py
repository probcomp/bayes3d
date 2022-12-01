import jax.numpy as jnp
from .transforms_3d import add_homogenous_ones

def camera_rays_from_params(height, width, fx, fy, cx, cy):
    rows, cols = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    pixel_coords = jnp.stack([rows,cols],axis=-1)
    pixel_coords_dir = (pixel_coords - jnp.array([cx,cy])) / jnp.array([fx,fy])
    pixel_coords_dir_h = add_homogenous_ones(pixel_coords_dir)
    return pixel_coords_dir_h