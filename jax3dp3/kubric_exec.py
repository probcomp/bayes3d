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
from custom_cam import KubCamera
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.core.color import get_color

#unpacking the data from the npz file
data_file = "/tmp/blenderproc_kubric.npz"
data = np.load(data_file, allow_pickle=True)
mesh_paths = data["mesh_paths"]
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

logging.basicConfig(level="INFO")

# --- create scene and attach a renderer to it
scene = kb.Scene(resolution=(width.item(), height.item()))
scene.ambient_illumination = get_color("red")
renderer = KubricRenderer(scene)
camera = KubCamera(name="camera", position=(0, 0, 0),
                              look_at=(0, 0, 1))
camera.update_intrinsic_values(height, width, fx.item(), fy.item(), cx.item(), cy.item(), near.item(), far.item())
scene += camera
scene += kb.DirectionalLight(
    name="sun", position=(0, -0.0, 0),
    look_at=(0, 0, 1), intensity=10.5
)

for i in range(len(mesh_paths)):
    obj = kb.FileBasedObject(
        asset_id=f"{i}", 
        render_filename=mesh_paths[i],
        simulation_filename=None,
        position=poses[i][0],
        scale=scaling_factor,
        quaternion=poses[i][1],
    )
    scene += obj

# --- render (and save the blender file)
# renderer.save_state("helloworld.blend")
frame = renderer.render_still()
np.savez("/tmp/output.npz", rgba=frame["rgba"], segmentation=frame["segmentation"], depth=frame["depth"])