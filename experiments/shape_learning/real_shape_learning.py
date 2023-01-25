
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

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

# filename = "panda_dataset/scene_4.pkl"
# filename = "panda_dataset_2/utensils.pkl"
# full_filename = "blue.pkl"
full_filename = "1674620488.514845.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
file = open(full_filename,'rb')
camera_image = pickle.load(file)
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


online_state = jax3dp3.Jax3DP3Perception()


depth = camera_image.depthPixels
rgb_original = camera_image.rgbPixels
K = camera_image.camera_matrix
# camera_pose = camera_image.camera_pose
camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)

# 1m, +-.6

# gripper_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
# camera_pose = (gripper_pose @ 
#     # t3d.inverse_pose(gripper_pose_to_cam_pose)
#     gripper_pose_to_cam_pose
# )


orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
orig_h,orig_w = depth.shape
orig_h,orig_w = depth.shape
near = 0.001
far = 4.99


online_state.set_camera_params(
    orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, near, far,
    scaling_factor=0.3
)

online_state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.1, 0.07, 0.04, 0.02],
    grid_params=[(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20)],
    likelihood_r_sched = [0.2, 0.15, 0.1, 0.04, 0.02]
)

point_cloud_image = online_state.process_depth_to_point_cloud_image(depth)
online_state.infer_table_plane(point_cloud_image, camera_pose)


point_cloud_image = online_state.process_depth_to_point_cloud_image(
    depth
)


point_cloud_image_above_table, segmentation_image  = online_state.segment_scene(
    rgb_original, point_cloud_image, camera_pose, "dashboard.png"
)

# jax3dp3.setup_visualizer()

# jax3dp3.show_cloud("c1",t3d.apply_transform(t3d.point_cloud_image_to_points(point_cloud_image_above_table), camera_pose))

from IPython import embed; embed()


