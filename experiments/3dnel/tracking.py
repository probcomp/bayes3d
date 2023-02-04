import matplotlib.pyplot as plt
import numpy as np
import os
from jax3dp3.viz import make_gif_from_pil_images
from PIL import Image
from copy import copy
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp

import sys
import cv2
import collections
import heapq
import sys

import warnings
import trimesh
import jax


top_level_dir = jax3dp3.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]
poses = data["poses"]
poses = jnp.array(poses)

cam_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2),
    jnp.array([0.0, -500.0, 0.0])
)
poses = t3d.inverse_pose(cam_pose) @ poses


state =jax3dp3.OnlineJax3DP3()
state.set_camera_parameters(data["camera_params"], scaling_factor=0.3)
(h,w,fx,fy,cx,cy,near,far) = state.camera_params

state.start_renderer()
for name in model_names:
    path = os.path.join(name, "textured.obj")
    state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )

point_cloud_images = jnp.array([
    state.process_depth_to_point_cloud_image(d)
    for d in depth_images
])
point_cloud_images = point_cloud_images.at[point_cloud_images[:,:,:,2] > far - 0.1,:].set(0.0)
jax3dp3.viz.save_depth_image(point_cloud_images[0,:,:,2], "depth.png", max=far)


images = jax3dp3.render_multiobject(poses, [0,1])
jax3dp3.viz.save_depth_image(images[:,:,2], "depth2.png", max=far)

deltas = jax3dp3.make_translation_grid_enumeration(
    -50.0, -50.0, -50.0,
    50.0, 50.0, 50.0,
    11,11,11
)
r, outlier_prob, outlier_volume = 0.05, 0.01, 1**3

initial_pose_estimates = poses.copy()

variances = jnp.diag(jnp.array([400.0, 100.0, 0.001]))
keys = jax.random.split(jax.random.PRNGKey(3), deltas.shape[0])
drift_poses = jax.vmap(jax3dp3.distributions.gaussian_pose, in_axes=(0,None))(keys, variances)

all_particles = []
viz_images = []
particles = jnp.tile(initial_pose_estimates[:, None, :,:], (1, deltas.shape[0], 1,1))
for t in range(point_cloud_images.shape[0]):
    obj_idx = 1
    particles = particles.at[1,:].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[1,:]))
    images = jax3dp3.render_multiobject_parallel(particles, [0,1])
    weights = jax3dp3.threedp3_likelihood_parallel_jit(
        point_cloud_images[t] / 1000.0, images / 1000.0, r, outlier_prob, outlier_volume
    )
    parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
    particles = particles[:,parent_idxs]
    keys = jax.random.split(keys[0], weights.shape[0])

    particle_image = jax3dp3.render_point_cloud(particles[1, :, :3, -1], h, w, fx,fy, cx,cy, near, far, 0)
    
    
    reconstruction = jax3dp3.viz.get_depth_image(point_cloud_images[t][:,:,2],  max=far)
    rgb =  jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_images[t],255.0),h,w)
    viz_images.append(
        jax3dp3.viz.multi_panel([
            rgb,
            jax3dp3.viz.get_depth_image(images[parent_idxs[0],:,:,2],max=far),
            jax3dp3.viz.get_depth_image(particle_image[:,:,2] > 0.0,max=2.0),
            reconstruction,
            jax3dp3.viz.overlay_image(rgb, reconstruction),
        ],
        labels=[
            "RGB",
            "Depth",
            "Particles",
            "Reconstruction",
            "Overlay",
        ])
    )
    all_particles.append(particles)

jax3dp3.viz.make_gif(viz_images, "out.gif")


from IPython import embed; embed()








# state.infer_table_plane(point_cloud_image, observation.camera_pose)
