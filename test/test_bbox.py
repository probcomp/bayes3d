import sys
sys.path.append('.')

import jax.numpy as jnp
from jax3dp3.viz import get_depth_image
from jax3dp3.bbox import overlay_bounding_box, proj_2d

h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
fx, fy = fx_fy
cx, cy = cx_cy   
max_depth = 5.0
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

observed = jnp.zeros((100,100,4))
depth_img = get_depth_image(observed[:,:,2], 5.0)

gx,gy,gz = 0.5, 0.25, 1
gridsize = 1.0

proj_2d(jnp.array([gx, gy, gz, 1]), K)

img = overlay_bounding_box(gx,gy,gz, gridsize, depth_img, K, save=True).resize((w*2, h*2))

img.save("bbox_test.png")
