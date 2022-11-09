import sys
import os
sys.path.append('.')

import jax
import jax.numpy as jnp
from jax3dp3.likelihood import sample_cloud_within_r, sample_coordinate_within_r
from jax3dp3.rendering import render_planes, render_cloud_at_pose
from jax3dp3.shape import get_cube_shape 
from jax3dp3.utils import depth_to_coords_in_camera
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
import matplotlib.pyplot as plt

#--------- Camera viz settings ------------
h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)
pixel_smudge = 0
max_depth = 5.0
K = jnp.array([[fx_fy[0], 0, cx_cy[0]], [0, fx_fy[1], cx_cy[1]], [0,0,1]])

cube_length = 0.5
shape = get_cube_shape(cube_length)


# -------- Figure settings ----------------
middle_width = 20
top_border = 45
num_images = 2  # gt and smapled

og_width = num_images * w + (num_images - 1) * middle_width
og_height = h + top_border

width_scaler = 2
height_scaler = 2

# --------- Render settings ---------------
render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)

sample_cloud_within_r_jit = jax.jit(sample_cloud_within_r)
# ------- GT rendering --------------------
pose_center = jnp.array([
    [1.0, 0.0, 0.0, 0],   
    [0.0, 1.0, 0.0, 0],   
    [0.0, 0.0, 1.0, 2],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
) 
gt_depth_img = get_depth_image(render_planes_jit(pose_center)[:,:,2], 5.0).resize((w*width_scaler, h*height_scaler))
save_depth_image(render_planes_jit(pose_center)[:,:,2], 5.0,"GTIMAGE.png")

pose_hypothesis_depth = render_planes_lambda(pose_center)[:, :, 2]  
cloud = depth_to_coords_in_camera(pose_hypothesis_depth, K)[0]

# -------Generate samples -----------------
min_r = 0
max_r = cube_length * 2

all_images = []
for likelihood_r in reversed(jnp.linspace(min_r, max_r, 40)):
    print("likelihood r =", likelihood_r)
    sampled_cloud_r = sample_cloud_within_r(cloud, likelihood_r)
    rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx_fy, cx_cy, pixel_smudge)
    
    hypothesis_depth_img = get_depth_image(rendered_cloud_r[:, :, 2], max_depth).resize((w*width_scaler, h*height_scaler))


    images = [gt_depth_img, hypothesis_depth_img]
    labels = [f"GT Image\ncube length={cube_length}", f"Likelihood evaluation\nr={int(likelihood_r*10000)/10000}"]
    dst = multi_panel(images, labels, middle_width*width_scaler, top_border*height_scaler, 20)
    all_images.append(dst)

# "pause" at last frame
for _ in range(5):
    all_images.append(dst)

    
all_images[0].save(
    fp=f"likelihood_out.gif",
    format="GIF",
    append_images=all_images,
    save_all=True,
    duration=200,
    loop=0,
)

cloud.shape


from IPython import embed; embed()
