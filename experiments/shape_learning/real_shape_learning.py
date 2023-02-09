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


full_filename = "blue.pkl"

file = open(full_filename,'rb')
data = pickle.load(file)
file.close()



calibration_data = {
  "qw": 0.7074450620054885,
  "qx": -0.06123049355593103,
  "qy": -0.05318248430843586,
  "qz": 0.7020996612061071,
  "x": 0.04268000721548824,
  "y": -0.01696075177674074,
  "z": 0.06000526018408979,
}

translation = jnp.array([
    calibration_data["x"], calibration_data["y"], calibration_data["z"]
])

R = t3d.xyzw_to_rotation_matrix(jnp.array([
    calibration_data["qx"], calibration_data["qy"], calibration_data["qz"],calibration_data["qw"],
]))
gripper_pose_to_cam_pose = t3d.transform_from_rot_and_pos(R, translation)


observations = []
for d in data:
    depth = np.array(d["depth"] / 1000.0) 
    camera_pose = t3d.pybullet_pose_to_transform(d["extrinsics"]) * gripper_pose_to_cam_pose
    rgb = np.array(d["rgb"])
    K = d["intrinsics"][0]
    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = depth.shape
    near = 0.001
    far = 4.99
    observation = jax3dp3.Jax3DP3Observation(rgb, depth, camera_pose, h,w,fx,fy,cx,cy,near,far)
    observations.append(observation)

state = jax3dp3.OnlineJax3DP3()
state.setup_on_initial_frame(observations[0], [],[])

clouds = []
clouds_in_world_frame = []
for observation in observations:
    obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
    cloud = t3d.point_cloud_image_to_points(obs_point_cloud_image)
    clouds.append(cloud)
    clouds_in_world_frame.append(t3d.apply_transform(cloud, observation.camera_pose))
fused_cloud = jnp.vstack(clouds_in_world_frame)


jax3dp3.viz.get_rgb_image(observation.rgb,255.0).save("rgb.png")
jax3dp3.viz.get_depth_image(observation.depth,max=far).save("depth.png")

jax3dp3.setup_visualizer()
jax3dp3.show_cloud("1", clouds[0])