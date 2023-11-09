mport numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os
import glob
import bayes3d.neural
import pickle
# Can be helpful for debugging:
# jax.config.update('jax_enable_checks', True) 
from bayes3d.neural.segmentation import carvekit_get_foreground_mask
import genjax

b.setup_visualizer()

### Load image

with open("img.pickle", "rb") as file:
    img = pickle.load(file)

K = img['camera_matrix']
fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
h,w = img["depthPixels"].shape
intrinsics = b.Intrinsics(h,w,fx,fy,cx,cy,0.001,10000.0)

rgbd = b.RGBD(
    img["rgbPixels"],
    img["depthPixels"],
    b.t3d.pybullet_pose_to_transform(img["camera_pose"]),
    intrinsics
)

### Crop out the table

def find_plane(point_cloud, threshold,  minPoints=100, maxIteration=1000):
    """
    Returns the pose of a plane from a point cloud.
    """
    plane = pyransac3d.Plane()
    plane_eq, inliers = plane.fit(point_cloud, threshold, minPoints=minPoints, maxIteration=maxIteration)
    plane_pose = b.utils.plane_eq_to_plane_pose(plane_eq)
    return plane_pose, inliers

scaling_factor = 0.3
rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd, scaling_factor)

cloud = b.unproject_depth(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics).reshape(-1,3)
too_big_indices = np.where(cloud[:,2] > 3.0)
cloud = cloud.at[too_big_indices, :].set(np.nan)

table_pose, inliers = find_plane(np.array(cloud), 0.01)

# table_pose = b.utils.find_plane(
#     np.array(cloud), 0.01
# )
face_child = 3

camera_pose = jnp.eye(4)
table_pose_in_cam_frame = b.t3d.inverse_pose(camera_pose) @ table_pose
if table_pose_in_cam_frame[2,2] > 0:
    table_pose = table_pose @ b.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)

obs_img = b.unproject_depth_jit(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics)

# set inliers to zero
x_indices, y_indices = np.unravel_index(inliers, obs_img.shape[:2])
obs_img2 = obs_img.at[x_indices, y_indices, :].set(np.nan)#jnp.array([100.0,100.0,100.0]))

x_indices, y_indices = np.unravel_index(too_big_indices, obs_img.shape[:2])
obs_img3 = obs_img2.at[x_indices, y_indices, :].set(np.nan)

b.clear()
b.show_cloud("obs3", obs_img3.reshape(-1,3))

### Setup renderer
b.setup_renderer(rgbd_scaled_down.intrinsics)
model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
for model_path in glob.glob(os.path.join(model_dir, "*.ply")):
    b.RENDERER.add_mesh_from_file(model_path, scaling_factor=1.0/1000.0)