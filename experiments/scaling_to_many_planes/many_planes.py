import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2

h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)

gt_pose = jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

shape = get_cube_shape(0.5)

render_planes_jit = jax.jit(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy))
gt_image = render_planes_jit(gt_pose)
print("gt_image.shape", gt_image.shape)


start = time.time()
i = render_planes_jit(gt_pose)
end = time.time()
print ("Time elapsed:", end - start)

render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))


num_images_sweep = jnp.arange(1, 552, 50)
times = []
for num_images in num_images_sweep:
    gt_poses = jnp.stack([gt_pose for _ in range(num_images)])
    i = render_planes_parallel_jit(gt_poses)

    start = time.time()
    i = render_planes_parallel_jit(gt_poses)
    end = time.time()
    print ("Time elapsed:", end - start)
    print ("Time per frame:", (end - start) / gt_poses.shape[0])
    times.append((end - start) / gt_poses.shape[0])
    print(num_images, times[-1])

plt.clf()
plt.plot(num_images_sweep, times, linewidth=4)
plt.xlabel("Number of Images Rendered in Batch")
plt.ylabel("Average Time per Image (s)")
plt.savefig("scaling_with_parallel_rendering.png")



shape = get_cube_shape(0.5)
planes = shape[0]
plane_dims = shape[1]

num_planes_multiplier_sweep = jnp.arange(1, 100, 4)
times = []
for num_planes_multiplier in num_planes_multiplier_sweep:
    shape = (
        np.vstack([planes for _ in range(num_planes_multiplier)]), 
        np.vstack([plane_dims for _ in range(num_planes_multiplier)])
    )
    render_planes_jit = jax.jit(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy))
    i = render_planes_jit(gt_pose)

    start = time.time()
    i = render_planes_jit(gt_pose)
    end = time.time()
    print ("Time elapsed:", end - start)
    times.append(end - start)
    print(num_planes_multiplier, times[-1])

plt.clf()
plt.plot(num_planes_multiplier_sweep, times, linewidth=4)
plt.xlabel("Number of Planes")
plt.ylabel("Time (s)")
plt.savefig("scaling_wih_number_of_planes.png")
