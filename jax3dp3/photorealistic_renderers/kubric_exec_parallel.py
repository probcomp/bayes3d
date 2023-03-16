# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.core.color import get_color

#unpacking the data from the npz file
data_file = "/tmp/blenderproc_kubric.npz"
data = np.load(data_file, allow_pickle=True)
mesh_paths = data["mesh_paths"]
print(mesh_paths)
poses = data["poses"]
camera_pose = data["camera_pose"]
K = data["K"]
height = data["height"]
width = data["width"]
scaling_factor = data["scaling_factor"]
fx = data["fx"]
fy = data["fy"]
cx = data["cx"]
cy = data["cy"]
near = data["near"]
far = data["far"]
intensity = float(data["intensity"])

logging.basicConfig(level="INFO")

#convert intrinsics to focal_length, sensor_width
focal_length = fx / width
sensor_width = 1 
print(f"POSES: {poses}")
print(f"CAMERA POSE: {camera_pose}")



for scene_number in range(len(poses)):
    # --- create scene and attach a renderer to it
    scene = kb.Scene(resolution=(width.item(), height.item()))
    # scene.ambient_illumination = get_color("red")
    renderer = KubricRenderer(scene)
    # --- create perspective camera 
    scene += kb.PerspectiveCamera(name="camera",
        position =camera_pose[0],quaternion=camera_pose[1], focal_length=focal_length, sensor_width=sensor_width)
    scene += kb.DirectionalLight(
        name="sun", position=(0, -0.0, 0),
        look_at=(0, 0, 1), intensity=intensity
    )

    for obj_number in range(len(poses[scene_number])):
        obj = kb.FileBasedObject(
            asset_id=f"1", 
            render_filename=mesh_paths[obj_number],
            simulation_filename=None,
            scale=scaling_factor,
            position=poses[scene_number][obj_number][0],
            quaternion=poses[scene_number][obj_number][1],
        )
        scene += obj

    frame = renderer.render_still()
    print(f"RENDERED FRAME ")
    np.savez(f"/tmp/{scene_number}.npz", rgba=frame["rgba"], segmentation=frame["segmentation"], depth=frame["depth"])
    print(f"SAVED FRAME USING NP")



