import machine_common_sense as mcs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from jax3dp3.viz.img import save_depth_image,save_rgb_image
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
import cv2
import jax.numpy as jnp

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_imgs = data["depth_images"]
seg_images = data["seg_images"]

scaling_factor = 0.25

fx = data["fx"] * scaling_factor
fy = data["fy"] * scaling_factor

cx = data["cx"] * scaling_factor
cy = data["cy"] * scaling_factor

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])
original_height = data["height"]
original_width = data["width"]
h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)


coord_images = []
for d in depth_imgs:
    coord_images.append(
        depth_to_coords_in_camera(cv2.resize(d.copy(), (w,h),interpolation=1), K.copy())[0]
    )
coord_images = np.stack(coord_images)
# coord_images[coord_images[:,:,:,2] > 40.0] = 0.0
# coord_images[coord_images[:,:,:,1] > 0.85,:] = 0.0
coord_images = np.concatenate([coord_images, np.ones(coord_images.shape[:3])[:,:,:,None] ], axis=-1)
fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])
coord_images = jnp.array(coord_images)


t = 20
d = depth_images[t]
