
import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
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

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/lego_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/spoon_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")

file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]

images = [j.RGBD.construct_from_camera_image(c) for c in camera_images]
intrinsics = j.camera.scale_camera_parameters(images[0].intrinsics, 0.3)
renderer = j.Renderer(intrinsics)

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
for (path, n) in zip(model_paths, mesh_names):
    renderer.add_mesh(j.mesh.center_mesh(trimesh.load(path)),n)

image = images[0]
j.run_classification(image, renderer)

j.run_occlusion_search(image, renderer, 3)

from IPython import embed; embed()



from IPython import embed; embed()

