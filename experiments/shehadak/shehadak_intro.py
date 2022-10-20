import os
import numpy as np
from PIL import Image
from jax3dp3.viz.img import save_depth_image
from jax3dp3.utils import depth_to_coords_in_camera
import jax.numpy as jnp

num_frames = 103
data_path = "data/videos"

width =  300
height =  300
fx =  150
fy =  150
cx =  150
cy =  150

fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx, cy])

# near =  0.01
# far =  100000

rgb_images, depth_images = [], []
rgb_images_pil = []
for i in range(num_frames):
    rgb_path = os.path.join(data_path, f"frames/frame_{i}.jpeg")
    rgb_img = Image.open(rgb_path)
    rgb_images_pil.append(rgb_img)
    rgb_images.append(np.array(rgb_img))

    depth_path = os.path.join(data_path, f"depths/frame_{i}.npy")
    depth_npy = np.load(depth_path)
    depth_images.append(depth_npy)

rgb_images_pil[0].save("rgb.png")
save_depth_image(depth_images[0], 30.0, "depth.png")

K = jnp.array([
    [fx_fy[0], 0.0, cx_cy[0]],
    [0.0, fx_fy[1], cx_cy[1]],
    [0.0, 0.0, 1.0],
])
coord_image,_ = depth_to_coords_in_camera(depth_images[0], K)
print(coord_image.shape)
# -.5 < x < 1
# -.5 < y < .5
# 1.2 < z < 4
mask = np.invert(
    (coord_image[:,:,0] < 1.0) *
    (coord_image[:,:,0] > -0.5) *
    (coord_image[:,:,1] < 0.28) *
    (coord_image[:,:,1] > -0.5) *
    (coord_image[:,:,2] < 4.0) *
    (coord_image[:,:,2] > 1.2) 
)
coord_image[mask,:] = 0.0
save_depth_image(coord_image[:,:,2], 30.0, "coord_image.png")



from IPython import embed; embed()


