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
import pybullet_planning
import cv2
import collections
import heapq
import sys

import warnings
import trimesh

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

top_level_dir = jax3dp3.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]
model_paths = [
    os.path.join(top_level_dir,"003_cracker_box/textured_simple.obj"),
    os.path.join(top_level_dir,"004_sugar_box/textured_simple.obj"),
]


data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]
poses = data["poses"]
poses = jnp.array(poses)

cam_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2),
    jnp.array([0.0, -0.5, 0.0])
)
poses = t3d.inverse_pose(cam_pose) @ poses

(h,w,fx,fy,cx,cy,near,far) = data["camera_params"]

state =jax3dp3.OnlineJax3DP3()
state.set_camera_parameters((h,w,fx,fy,cx,cy,near,far), scaling_factor=0.3)

state.start_renderer()
for (name, path) in zip(model_names, model_paths):
    state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )


point_cloud_images = jnp.array([
    state.process_depth_to_point_cloud_image(d)
    for d in depth_images
])
jax3dp3.viz.save_depth_image(point_cloud_images[0,:,:,2], "depth.png", max=3.0)


images = jax3dp3.render_multiobject(poses, [0,1])
jax3dp3.viz.save_depth_image(images[:,:,2], "depth2.png", max=3.0)

deltas = jax3dp3.make_translation_grid_enumeration(
    -0.1, -0.1, -0.1,
    0.1, 0.1, 0.1,
    11,11,11
)


r, outlier_prob, outlier_volume = 0.01, 0.01, 1*3

viz_images = []

for t in range(point_cloud_images.shape[0]):
    pose_proposals = jnp.tile(poses[:, None, :,:], (1, deltas.shape[0], 1,1))
    pose_proposals = pose_proposals.at[1,:,:,:].set(jnp.einsum("aij,jk->aik", deltas, poses[1]))

    images = jax3dp3.render_multiobject_parallel(pose_proposals, [0,1])

    weights = jax3dp3.threedp3_likelihood_parallel_jit(
        point_cloud_images[t], images, r, outlier_prob, outlier_volume
    )
    best_idx = weights.argmax()
    best_delta = deltas[best_idx]
    poses = pose_proposals[:, best_idx]

    viz_images.append(
        jax3dp3.viz.vstack_images([
            jax3dp3.viz.get_depth_image(images[best_idx,:,:,2],max=3.0),
            jax3dp3.viz.get_depth_image(point_cloud_images[t][:,:,2],  max=3.0)
        ])
    )
jax3dp3.viz.make_gif(viz_images, "out.gif")


from IPython import embed; embed()








# state.infer_table_plane(point_cloud_image, observation.camera_pose)
