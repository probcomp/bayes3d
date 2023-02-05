import matplotlib.pyplot as plt
import numpy as np
import os
import pybullet as p
import pybullet_data
from jax3dp3.viz import make_gif_from_pil_images
from PIL import Image
from copy import copy
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import trimesh

import sys
import cv2
import collections
import heapq



p.connect(p.GUI)
# p.setGravity(0, 0, -5)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF

p.resetSimulation()

top_level_dir = jax3dp3.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]
model_paths = [
    os.path.join(top_level_dir,"003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"004_sugar_box/textured.obj"),
]

objects = []
box_dims = []
for (name, path) in zip(model_names,model_paths):
    mesh = trimesh.load(path)
    mesh = jax3dp3.mesh.center_mesh(mesh)
    mesh.vertices = mesh.vertices * 1000.0
    os.makedirs(name,exist_ok=True)
    mesh.export(os.path.join(name, "textured.obj"))

objects = []
box_dims = []
for name in model_names:
    path = os.path.join(name, "textured.obj")
    obj, dims = jax3dp3.pybullet.add_mesh(path)
    objects.append(obj)
    box_dims.append(dims)
# planeId = p.loadURDF("plane.urdf")

h, w, fx,fy, cx,cy = (
    480,
    640,
    500.0,500.0,
    320.0,240.0
)
near,far = 1.0, 2000.0



cracker_box_pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 100.0])) @ t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), -jnp.pi/2)
sugar_box_pose = t3d.transform_from_pos(jnp.array([-300.0, 500.0, 100.0])) @ t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), -jnp.pi/2)

cam_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2),
    jnp.array([0.0, -500.0, 100.0])
)

rgb_images = []
depth_images = []
for x in jnp.linspace(0.0, 600.0, 40):
    jax3dp3.pybullet.set_pose_wrapped(objects[0], cracker_box_pose)
    jax3dp3.pybullet.set_pose_wrapped(objects[1], t3d.transform_from_pos(jnp.array([x, 0.0, 0.0])) @ sugar_box_pose)

    rgb, depth, segmentation = jax3dp3.pybullet.capture_image(
        cam_pose,
        h, w, fx,fy, cx,cy , near, far
    )
    rgb_images.append(rgb)
    depth_images.append(depth)

np.savez("data.npz", rgb_images=rgb_images, depth_images=depth_images, camera_params=(h,w,fx,fy,cx,cy,near,far), poses=[cracker_box_pose, sugar_box_pose], camera_pose=cam_pose)
viz_images = [
    jax3dp3.viz.get_rgb_image(rgb, 255.0) for rgb in rgb_images
]
jax3dp3.viz.make_gif(viz_images,"rgb.gif")

from IPython import embed; embed()