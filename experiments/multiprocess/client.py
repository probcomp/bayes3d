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

step_metadatas = [step_metadata]
step_metadata = controller.step("Pass")
step_metadata = controller.step("Pass")
step_metadata = controller.step("Pass")

data_to_send = "aabbaa"


# Send and wait to receive.
socket.send(zlib.compress(pickle5.dumps(data_to_send)))
segments = pickle5.loads(zlib.decompress(socket.recv()))

print(segments)

step_metadatas = [step_metadata]
step_metadata = controller.step("Pass")
step_metadata = controller.step("Pass")
step_metadata = controller.step("Pass")

socket.send(zlib.compress(pickle5.dumps(data_to_send)))
segments = pickle5.loads(zlib.decompress(socket.recv()))
print(segments)