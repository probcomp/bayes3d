import logging
import os 
import random
import bpy
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
import numpy as np

logging.basicConfig(level="INFO")


data = np.load("/tmp/scene_data.npz")

# OBJ Paths
mesh_paths = data["mesh_paths"]
# 4x4 transformation matrices of the objects
poses = data["poses"]

camera_pose = data["camera_pose"]

# Intrinsics
K = data["K"]
height = data["height"]
width = data["width"]


scaling_factor = data["scaling_factor"]

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=resolution)
renderer = KubricRenderer(scene)

scene += kb.PerspectiveCamera(name="camera", position=(-8,-8, 3), look_at=(4, 4, 3))
scene += kb.DirectionalLight(name="sun", position=(0, 0, 5), look_at=(0, 0, 0), intensity=5)

# --- add objects to the scene
for (i, obj_path) in enumerate(mesh_paths):
    position, orientation = f(poses[i])

    obj = kb.FileBasedObject(
        asset_id=f"{i}", 
        render_filename=obj_path,
        simulation_filename=None,
        position=obj["position"],
        scale=obj["scale"],
        quaternion=obj["quaternion"],
    )
    scene += obj

# --- render (and save the blender file)
renderer.save_state(os.path.join(output_dir, scene_name + ".blend"))
frame = renderer.render_still()
kb.write_png(frame["rgba"], os.path.join(output_dir, scene_name + ".png"))
kb.write_palette_png(frame["segmentation"], os.path.join(output_dir, scene_name + "_segmentation.png"))
scale = kb.write_scaled_png(frame["depth"], os.path.join(output_dir, scene_name + "_depth.png"))
logging.info("Depth scale: %s", scale)