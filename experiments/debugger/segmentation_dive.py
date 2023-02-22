
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

sys.path.extend(["/home/ubuntu/ptamp/pybullet_planning"])
sys.path.extend(["/home/ubuntu/ptamp"])
warnings.filterwarnings("ignore")

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/utensils.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/cracker_sugar_banana_real.pkl")
test_pkl_name = test_pkl_file.split(".")[0].split("/")[-1]

file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]


file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

observation = observations[0]
state = jax3dp3.OnlineJax3DP3()

state.setup_on_initial_frame(observations[0], [], [])


obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
# segmentation_image, dashboard_viz = state.segment_scene(observation.rgb, obs_point_cloud_image)
segmentation_image, dashboard_viz = state.segment_scene(observation.rgb, obs_point_cloud_image, observation.depth, use_nn=True)


print(f"Saving image for {test_pkl_name}")
dashboard_viz.save(f"segmentation_final_{test_pkl_name}.png")

from IPython import embed; embed()





# state.step(observation, 1)









