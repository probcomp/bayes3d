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
import bpy
import os
from kubric import file_io

parser = kb.ArgumentParser()
parser.add_argument("--hdri_assets", type=str)

parser.add_argument("--kubasic_assets", type=str)

FLAGS = parser.parse_args()


#unpacking the data from the npz file
data_file = "/tmp/blenderproc_kubric.npz"
data = np.load(data_file, allow_pickle=True)
mesh_paths = data["mesh_paths"]
#print(mesh_paths)
obj_poses = data["obj_poses"]
camera_poses = data["camera_poses"]
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
dome_pose = data["dome_pose"]
frames = data["frames"]
seed = data["seed"]

print('im H: ' + str(height), 'im W: ' + str(width))
#print('scaling factor: ' + str(scaling_factor))
#scaling_factor = 1 # this is a hacked scale

logging.basicConfig(level="INFO")

#convert intrinsics to focal_length, sensor_width
focal_length = fx / width
sensor_width = 1 

print('focal length: ' + str(focal_length))

frame_start = 1
frame_end = int(frames)

# --- create scene and attach a renderer to it


#scene = kb.Scene(resolution=(width.item(), height.item()), frame_start=frame_start, frame_end=frame_end, frame_rate=24) 
scene = kb.Scene(resolution=(width.item(), height.item())) 
## Set up the camera
scene.camera = kb.PerspectiveCamera(name="camera", focal_length=focal_length, sensor_width=sensor_width)

# scene.ambient_illumination = get_color("red")
renderer = KubricRenderer(scene)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
# --- create perspective camera 

# scene += kb.PerspectiveCamera(name="camera",
#     position =camera_pose[0],quaternion=camera_pose[1], focal_length=focal_length, sensor_width=sensor_width)


scene += kb.DirectionalLight(
    name="sun", position=(0, 0.0, 2),
    look_at=(0, 0, 0), intensity=intensity
)


# --- Populate the scene
for obj_number in range(len(obj_poses)):
    obj = kb.FileBasedObject(
        asset_id=f"1", 
        render_filename=mesh_paths[obj_number],
        simulation_filename=None,
        scale=scaling_factor,
        position=obj_poses[obj_number][0],
        quaternion=obj_poses[obj_number][1],
    )
    scene += obj

# test turn off dome and background!

# # background HDRI
# hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
# train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)

# logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
# rng = np.random.RandomState(int(seed))
# hdri_id = rng.choice(train_backgrounds)

# background_hdri = hdri_source.create(asset_id=hdri_id)
# logging.info("Using background %s", hdri_id)
# scene.metadata["background"] = hdri_id
# renderer._set_ambient_light_hdri(background_hdri.filename)

# # Dome

# dome = kubasic.create(asset_id="dome", name="dome",
# #                    friction=1.0,
# #                    restitution=0.0,
#                     #static=True, #test commenting this if render still bad
#                     background=True, # maybe we set this to false????
#                     position=dome_pose[0], quaternion=dome_pose[1])
# assert isinstance(dome, kb.FileBasedObject)
# scene += dome
# dome_blender = dome.linked_objects[renderer]
# print(dome_blender.data.materials[0].node_tree.nodes["Image Texture"])
# texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
# texture_node.image = bpy.data.images.load(background_hdri.filename)

# ## Print bounds of dome: dome starts out with -40 to 40 bounding box, dome scale is sensitive due to clipping of the half-sphere dimensions
# print('dome bounds' + str(dome.bbox_3d))

# sf = 0.05
# dome.scale = np.array([sf, sf, sf])




## Render 

for frame in range(frame_start-1, frame_end):

    scene.camera.position = np.array(camera_poses[frame][0])
    scene.camera.quaternion = np.array(camera_poses[frame][1])

    fr = renderer.render_still()

    np.savez(f"/tmp/{frame}.npz", rgba=fr["rgba"], segmentation=fr["segmentation"], depth=fr["depth"],
                position=scene.camera.position, orientation=scene.camera.quaternion)
        


'''

data_stack = renderer.render(return_layers=["rgba", "segmentation", "depth"])

for scene_number in range(len(data_stack['rgba'])):
    np.savez(f"/tmp/{scene_number}.npz", rgba=data_stack["rgba"][scene_number], segmentation=data_stack["segmentation"][scene_number], depth=data_stack["depth"][scene_number],
             position=positions[scene_number], orientation=orientations[scene_number])

# np.savez(f"/tmp/{scene_number}.npz", rgba=frame["rgba"], segmentation=frame["segmentation"], depth=frame["depth"])

kb.done()
'''


## TODO: fix the object scalings and maybe generate more realistic trajectories
## more realistic trajectories meaning linear?


## fix the randomness to make random trajectories with controllable generation