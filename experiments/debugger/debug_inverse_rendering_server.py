
import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3
from dataclasses import dataclass
import sys
import warnings
import pybullet_planning
import cv2
import pickle
import zlib
import zmq

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

# filename = "panda_dataset/scene_4.pkl"
# filename = "panda_dataset_2/utensils.pkl"
# full_filename = "blue.pkl"
full_filename = "1674620488.514845.pkl"
full_filename = "knife_spoon.pkl"
full_filename = "demo2_nolight.pkl"
full_filename = "strawberry_error.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "knife_spoon.pkl"
full_filename = "utensils.pkl"
full_filename = "new_utencils.pkl"
full_filename = "utensils.pkl"
file = open(full_filename,'rb')
camera_image = pickle.load(file)["camera_images"][0]
file.close()

depth = np.array(camera_image.depthPixels) 
camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
rgb_original = np.array(camera_image.rgbPixels)
K = camera_image.camera_matrix
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
orig_h,orig_w = depth.shape
near = 0.001
far = 4.99

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:5554")

socket.send(zlib.compress(pickle.dumps({"timestep": 0, "camera_images": [camera_image]})))

segments = pickle.loads(zlib.decompress(socket.recv()))


from IPython import embed; embed()
