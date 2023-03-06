import jax3dp3 as j
import zmq
import pickle5
import zlib

import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import zlib
import pickle5


import zmq
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5554")

controller = mcs.create_controller(
    os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
)

scene_data = mcs.load_scene_json_file(os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "charlie_0002_04_B1_debug.json"))

step_metadata = controller.start_scene(scene_data)
image = j.RGBD.construct_from_step_metadata(step_metadata)



intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=50.0
)
renderer = j.Renderer(intrinsics)

