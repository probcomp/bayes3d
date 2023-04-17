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
import jax3dp3.pybullet_utils
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import trimesh

import sys
import cv2
import collections
import heapq

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF


import pybullet_planning
top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
]
meshes = [
    trimesh.load(path) for path in model_paths
]

objects = []
box_dims = []
for (name, path) in zip(model_names,model_paths):
    mesh = trimesh.load(path)
    mesh = jax3dp3.mesh.center_mesh(mesh)
    mesh.vertices = mesh.vertices
    os.makedirs(name,exist_ok=True)
    mesh.export(os.path.join(name, "textured.obj"))


p.resetSimulation()

table_obj, table_dims = jax3dp3.pybullet_utils.create_table(
    0.5,
    0.5,
    0.1,
    0.01,
    0.01,
)
objects = [table_obj]
box_dims = [table_dims]

for name in model_names:
    path = os.path.join(name, "textured.obj")
    obj, dims = jax3dp3.pybullet_utils.add_mesh(path)
    objects.append(obj)
    box_dims.append(dims)
box_dims = jnp.array(box_dims)

h, w, fx,fy, cx,cy = (
    480,
    640,
    500.0,500.0,
    320.0,240.0
)
near,far = 0.01, 5.0


start_poses = jnp.array([
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
])

edges = jnp.array([
    [-1,0],
    [0,1],
    [0,2],
    [0,3],
])

contact_params = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.3, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ]
)

face_parent = jnp.array([3,3,3,3])
face_child = jnp.array([2,2,2,2])

poses = jax3dp3.scene_graph.absolute_poses_from_scene_graph(
    start_poses, box_dims, edges, contact_params, face_parent, face_child
)

cam_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/4),
    jnp.array([0.0, -0.5, 0.4])
)

for (obj, p) in zip(objects, poses[1:]):
    jax3dp3.pybullet_utils.set_pose_wrapped(obj, p)

rgb, depth, segmentation = jax3dp3.pybullet_utils.capture_image(
    cam_pose,
    h, w, fx,fy, cx,cy , near, far
)
jax3dp3.viz.get_rgb_image(rgb,255.0).save("rgb.png")

rgb_images = []
depth_images = []
for x in jnp.linspace(0.0, 600.0, 40):

    rgb_images.append(rgb)
    depth_images.append(depth)

np.savez("data.npz", rgb_images=rgb_images, depth_images=depth_images, camera_params=(h,w,fx,fy,cx,cy,near,far), poses=[cracker_box_pose, sugar_box_pose], camera_pose=cam_pose)
viz_images = [
    jax3dp3.viz.get_rgb_image(rgb, 255.0) for rgb in rgb_images
]
jax3dp3.viz.make_gif(viz_images,"rgb.gif")

from IPython import embed; embed()