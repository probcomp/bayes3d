import cv2
import numpy as np
import matplotlib.pyplot as plt

import glob
import random
import sys


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
import jax3dp3


sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/tag.pkl")
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)

observations = [
    jax3dp3.Jax3DP3Observation.construct_from_aidan_dict(d) for d in camera_images
]

rgb_viz = []
for obs in observations:
    rgb_viz.append(
        jax3dp3.viz.get_rgb_image(obs.rgb, 255.0)
    )
jax3dp3.viz.multi_panel(rgb_viz).save("rgb.png")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((7, 9), 0.024, 0.016, aruco_dict)
detector = cv2.aruco.CharucoDetector(board)

images = []
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
for observation in observations:
    frame = observation.rgb
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    new_frame = cv2.aruco.drawDetectedCornersCharuco(
        frame, 
        charuco_corners, charuco_ids
    )
    images.append(
        jax3dp3.viz.get_rgb_image(new_frame, 255.0)
    )

    corners_all.append(charuco_corners)
    ids_all.append(charuco_ids)
jax3dp3.viz.multi_panel(images).save("detect.png")

image_size = gray.shape[::-1]
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

fx,fy,cx,cy = cameraMatrix[0,0], cameraMatrix[1,1],  cameraMatrix[0,2],  cameraMatrix[1,2]


jax3dp3.setup_visualizer()

board_poses = []
for (rvec, tvec) in zip(rvecs, tvecs):
    pose = t3d.transform_from_rvec_tvec(rvec,tvec)
    board_poses.append(pose)
camera_poses = [o.camera_pose for o in observations]
board_poses = jnp.array(board_poses)
camera_poses = jnp.array(camera_poses)

for (i,p) in enumerate(board_poses):
    jax3dp3.show_pose(f"a{i}", t3d.inverse_pose(p))

for (i,p) in enumerate(camera_poses):
    jax3dp3.show_pose(f"b{i}", p)

clouds = []
for (cam_pose, observation) in zip(camera_poses, observations):
    point_cloud_image = t3d.depth_to_point_cloud_image(observation.depth, fx, fy, cx,cy)
    clouds.append(t3d.apply_transform(
        t3d.point_cloud_image_to_points(point_cloud_image),
        cam_pose
    ))
import distinctipy        
colors = distinctipy.get_colors(len(clouds), pastel_factor=0.2)
jax3dp3.clear()
for i in range(len(clouds)):
    jax3dp3.show_cloud(f"{i}", clouds[i]*3.0, color=np.array(colors[i]))


Rot, trans = cv2.calibrateHandEye(
    np.array(camera_poses[:,:3,:3]),
    np.array(camera_poses[:,:3,3]),
    np.array(board_poses[:,:3,:3]),
    np.array(board_poses[:,:3,3]),
)
cam_in_gripper = t3d.transform_from_rot_and_pos(
    jnp.array(Rot),
    jnp.array(trans)
)


clouds = []
for (cam_pose, observation) in zip(camera_poses, observations):
    point_cloud_image = t3d.depth_to_point_cloud_image(observation.depth, fx, fy, cx,cy)
    clouds.append(t3d.apply_transform(
        t3d.point_cloud_image_to_points(point_cloud_image),
        cam_pose @ cam_in_gripper
    ))
import distinctipy        
colors = distinctipy.get_colors(len(clouds), pastel_factor=0.2)
jax3dp3.clear()
for i in range(len(clouds)):
    jax3dp3.show_cloud(f"{i}", clouds[i]*3.0, color=np.array(colors[i]))

print("Cam in Gripper")
print(cam_in_gripper)

camera_data_file = os.path.join(jax3dp3.utils.get_assets_dir(),"camera_data.pkl")
pickle.dump((cam_in_gripper, (fx,fy,cx,cy)), open(camera_data_file,"wb"))



from IPython import embed; embed()