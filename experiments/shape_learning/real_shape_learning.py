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


full_filename = "red_lego_multi.pkl"
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs", full_filename)
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)
file.close()

camera_data_file = os.path.join(jax3dp3.utils.get_assets_dir(),"camera_data.pkl")
gripper_to_cam, (fx,fy,cx,cy) = pickle.load(open(camera_data_file,"rb"))
gripper_pose_to_cam_pose = jnp.array(gripper_to_cam)

h,w = camera_images[0]["rgb"].shape[:2]


observations = [
    jax3dp3.Jax3DP3Observation.construct_from_aidan_dict(d) for d in camera_images
]
for i in range(len(observations)):
    observations[i].camera_pose = observations[i].camera_pose @ gripper_pose_to_cam_pose
    observations[i].camera_params = (h,w,fx,fy,cx,cy,0.01, 5.0)

rgb_viz = []
for obs in observations:
    rgb_viz.append(
        jax3dp3.viz.get_rgb_image(obs.rgb, 255.0)
    )
jax3dp3.viz.multi_panel(rgb_viz).save("rgb.png")


state = jax3dp3.OnlineJax3DP3()
state.setup_on_initial_frame(observations[0], [],[])
jax3dp3.setup_visualizer()

clouds = []
clouds_in_world_frame = []
for observation in observations:
    obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
    cloud = t3d.point_cloud_image_to_points(obs_point_cloud_image)
    clouds.append(cloud)
    clouds_in_world_frame.append(t3d.apply_transform(cloud, observation.camera_pose))
fused_cloud = jnp.vstack(clouds_in_world_frame)

import distinctipy
colors = distinctipy.get_colors(len(clouds_in_world_frame), pastel_factor=0.2)
jax3dp3.clear()
for i in range(len(clouds_in_world_frame)):
    jax3dp3.show_cloud(f"{i}", clouds_in_world_frame[i]*3.0, color=np.array(colors[i]))

state.learn_new_object(observations, 1)

from IPython import embed; embed()