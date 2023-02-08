
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
import collections
import heapq

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")


full_filename = "strawberry_error.pkl"
full_filename = "knife_sim.pkl"

full_filename = "demo2_nolight.pkl"
file = open(full_filename,'rb')
camera_images = pickle.load(file)["camera_images"]
file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

from IPython import embed; embed()

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

from IPython import embed; embed()
orig_h, orig_w = observations[0].camera_params[:2]

state = jax3dp3.OnlineJax3DP3()

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
# os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
# os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
# os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]

state.setup_on_initial_frame(
    observations[0],
    model_names,
    model_paths
)

state.step(
    observations[-1],
    1
)


from IPython import embed; embed()



