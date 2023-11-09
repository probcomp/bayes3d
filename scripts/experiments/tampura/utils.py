import numpy as np
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

import pyransac3d
def find_plane(point_cloud, threshold,  minPoints=100, maxIteration=1000):
    """
    Returns the pose of a plane from a point cloud.
    """
    plane = pyransac3d.Plane()
    plane_eq, inliers = plane.fit(point_cloud, threshold, minPoints=minPoints, maxIteration=maxIteration)
    plane_pose = b.utils.plane_eq_to_plane_pose(plane_eq)
    return plane_pose, inliers

def load_img(fname):
    with open(fname, "rb") as file:
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

    return rgbd

def scale_remove_and_setup_renderer(rgbd, scaling_factor = 0.5):    
    rgbd_scaled_down = b.RGBD.scale_rgbd(rgbd, scaling_factor)

    b.setup_renderer(rgbd_scaled_down.intrinsics)

    cloud = b.unproject_depth(rgbd_scaled_down.depth, rgbd_scaled_down.intrinsics).reshape(-1,3)
    too_big_indices = np.where(cloud[:,2] > 3.0)
    cloud = cloud.at[too_big_indices, :].set(np.nan)

    too_small_indices = np.where(cloud[:,2] < 0.1)
    cloud = cloud.at[too_small_indices, :].set(np.nan)

    table_pose, inliers = find_plane(np.array(cloud), 0.01)

    camera_pose = jnp.eye(4)
    table_pose_in_cam_frame = b.t3d.inverse_pose(camera_pose) @ table_pose
    if table_pose_in_cam_frame[2,2] > 0:
        table_pose = table_pose @ b.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)

    # Depth image, cropping out table, too far, and too close points
    depth_im = jnp.array(rgbd_scaled_down.depth)
    x_indices, y_indices = np.unravel_index(inliers, depth_im.shape) 
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)
    x_indices, y_indices = np.unravel_index(too_big_indices, depth_im.shape)
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)
    x_indices, y_indices = np.unravel_index(too_small_indices, depth_im.shape)
    depth_im = depth_im.at[x_indices, y_indices].set(b.RENDERER.intrinsics.far)

    obs_img = b.unproject_depth_jit(depth_im, rgbd_scaled_down.intrinsics)

    return rgbd_scaled_down, obs_img, table_pose, cloud, depth_im

def add_meshes_to_renderer():
    # Add every ply file from the model dir using b.RENDERER.add_mesh_from_file(mesh_path)
    model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
    for model_path in glob.glob(os.path.join(model_dir, "*.ply")):
        b.RENDERER.add_mesh_from_file(model_path, scaling_factor=1.0/1000.0)

### Grid-based inference ###
def get_grids(param_sequence):
    return [
        b.utils.make_translation_grid_enumeration_3d(
            -x, -x, -ang, x, x, ang, *nums
        ) for (x, ang, nums) in param_sequence
    ]

def c2f(
    # n = num objects
    table_pose, # 4x4 pose
    face_child,
    potential_cps, # (n, 3)
    potential_indices, # (n,)
    number, # = n - 1
    contact_param_gridding_schedule,
    obs_img
):
    for cp_grid in contact_param_gridding_schedule:
        potential_cps, score = grid_and_max(table_pose, face_child, potential_cps, potential_indices, number, cp_grid, obs_img)
    return potential_cps, score
c2f_jit = jax.jit(c2f)

def grid_and_max(
    # n = num objects; g = num grid points
    table_pose, 
    face_child,
    cps, # (n, 3)
    indices, # (n,)
    number,
    grid,
    obs_img
):
    cps_expanded = jnp.repeat(cps[None,...], grid.shape[0], axis=0) # (g, n, 3)
    cps_expanded = cps_expanded.at[:,number,:].set(cps_expanded[:,number,:] + grid) # (g, n, 3)
    cp_poses = cps_to_pose_parallel(cps_expanded, indices, table_pose, face_child) # (g, n, 4, 4)
    rendered_images = b.RENDERER.render_many(cp_poses, indices)[...,:3] # (g, h, w, 3)
    # I think the 3 dimension is an xyz point in the camera frame(?)

    scores = score_vmap(rendered_images, obs_img)
    best_idx = jnp.argmax(scores) # jnp.argsort(scores)[-4]
    cps = cps_expanded[best_idx]
    return cps, scores[best_idx]

def _cp_to_pose(cp, index, table_pose, face_child):
    return table_pose @ b.scene_graph.relative_pose_from_edge(cp, face_child, b.RENDERER.model_box_dims[index])
cps_to_pose = jax.vmap(_cp_to_pose, in_axes=(0,0,None,None))
cps_to_pose_parallel = jax.vmap(cps_to_pose, in_axes=(0,None,None,None))

def score_images(
    rendered, # (h, w, 3) - point cloud
    observed # (h, w, 3) - point cloud
):
    # get L2 distance between each corresponding point
    distances = jnp.linalg.norm(observed - rendered, axis=-1)
    width = 0.04

    # Contribute 1/(h*w) * 1/width to the score for ach nearby pixel,
    # and contribute nothing for each faraway pixel.
    vals = (distances < width/2) / width
    return vals.mean()

score_vmap = jax.jit(jax.vmap(score_images, in_axes=(0, None)))

####

# points where observed is [0, 0, 0]
# nonzero_vals = (observed != np.array([0., 0., 0.])).any(axis=-1)
# nonzero_vals = jnp.where(observed != jnp.array([0., 0., 0.]), axis=-1)

# rendered_dists = jnp.linalg.norm(rendered, axis=-1)
# near_rendered_pts = (rendered_dists < 1000.) # (h, w)
# near_rendered_pts = ((rendered < 100.) * (rendered > -100.)).all(axis=-1) # (h, w)

# probabilities_per_pixel = (distances < width/2) / width
# return probabilities_per_pixel.mean()
    # penalize large distances, for pixels not at the max depth of the renderer
    # vals = vals - ( (distances > 2*width) * near_rendered_pts  / width )
