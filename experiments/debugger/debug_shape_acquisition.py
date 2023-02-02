
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

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

# filename = "panda_dataset/scene_4.pkl"
# filename = "panda_dataset_2/utensils.pkl"
# full_filename = "blue.pkl"
full_filename = "1674620488.514845.pkl"
full_filename = "knife_spoon.pkl"
full_filename = "demo2_nolight.pkl"
full_filename = "utensils.pkl"
full_filename = "strawberry_error.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "knife_spoon.pkl"
full_filename = "new_utencils.pkl"
full_filename = "blue_more.pkl"
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


from collections import namedtuple
CameraImage = namedtuple("CameraImage", ["depthPixels", "rgbPixels","camera_pose","camera_matrix"])
camera_image = CameraImage(
    camera_image[0]["depth"] / 1000.0,
    camera_image[0]["rgb"],
    camera_image[0]["extrinsics"],
    camera_image[0]["intrinsics"][0]
)



depth = np.array(camera_image.depthPixels) 
camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose) * gripper_pose_to_cam_pose
rgb_original = np.array(camera_image.rgbPixels)
K = camera_image.camera_matrix
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
orig_h,orig_w = depth.shape
near = 0.001
far = 4.99

online_state = jax3dp3.OnlineJax3DP3(
    orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, near, far,
    scaling_factor=0.3
)


(h,w,fx,fy,cx,cy, near, far) = online_state.camera_params
rgb_scaled = jax3dp3.utils.resize(rgb_original,h,w)
hsv_scaled = cv2.cvtColor(rgb_scaled, cv2.COLOR_RGB2HSV)

gray_colors = [jnp.array([120, 120, 120])]
error_cumulative = jnp.ones((h,w))
for gray in gray_colors:
    errors = jnp.abs(rgb_scaled[:,:,:] - gray_colors).sum(-1)
    # value_thresh = hsv_scaled[:,:,-1] < 75.0
    # value_thresh2 = hsv_scaled[:,:,-1] > 175.0
    # error_cumulative *=  np.logical_or((errors > 200), value_thresh, value_thresh2)
    error_cumulative *=  (errors > 150)

# gray_colors = [jnp.array([158, 9])]
# error_cumulative = jnp.ones((h,w))
# for gray in gray_colors:
#     errors = jnp.abs(hsv_scaled[:,:,:2] - gray).sum(-1)
#     # value_thresh = hsv_scaled[:,:,-1] < 75.0
#     # value_thresh2 = hsv_scaled[:,:,-1] > 175.0
#     # error_cumulative *=  np.logical_or((errors > 200), value_thresh, value_thresh2)
#     error_cumulative *=  (errors > 140)

gray_mask = error_cumulative

point_cloud_image_pre_remove_table = online_state.process_depth_to_point_cloud_image(depth)
jax3dp3.viz.save_depth_image(point_cloud_image_pre_remove_table[:,:,2]  * gray_mask,"depth.png",max=far)

online_state.infer_table_plane(point_cloud_image_pre_remove_table, camera_pose)
point_cloud_image = point_cloud_image_pre_remove_table * gray_mask[:,:,None]
jax3dp3.viz.save_depth_image(point_cloud_image[:,:,2],"depth.png",max=far)


point_cloud_image_above_table, segmentation_image  = online_state.segment_scene(
    rgb_original, point_cloud_image, camera_pose, "dashboard.png",
    FLOOR_THRESHOLD=-0.01,
    TOO_CLOSE_THRESHOLD=0.4,
    FAR_AWAY_THRESHOLD=0.8,
)

jax3dp3.setup_visualizer()

jax3dp3.show_cloud("c1", t3d.apply_transform(
    t3d.point_cloud_image_to_points(point_cloud_image_above_table),
    camera_pose
    )
)


from IPython import embed; embed()

# jax3dp3.setup_visualizer()

# jax3dp3.show_cloud("c1",t3d.apply_transform(
#     t3d.point_cloud_image_to_points(point_cloud_image), jnp.linalg.inv(online_state.table_pose) @ camera_pose) )



# jax3dp3.setup_visualizer()

# jax3dp3.show_cloud("c1",t3d.apply_transform(t3d.point_cloud_image_to_points(point_cloud_image_above_table), camera_pose))

from IPython import embed; embed()


top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]


for (name, path) in zip(model_names, model_paths):
    online_state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )

online_state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.1, 0.07, 0.04, 0.02],
    grid_params=[(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20)],
    likelihood_r_sched = [0.2, 0.15, 0.1, 0.04, 0.02]
)

    

outlier_prob, outlier_volume = 0.2, 10**3

unique =  np.unique(segmentation_image)
segmentation_idxs_to_do_pose_estimation_for = unique[unique != -1]
all_results = []
for seg_id in segmentation_idxs_to_do_pose_estimation_for:
    results = online_state.run_detection(
        rgb_original,
        point_cloud_image,
        point_cloud_image_above_table,
        segmentation_image,
        seg_id,
        camera_pose,
        outlier_prob,
        outlier_volume,
        0.06,
        0.07,
        f"out__seg_id_{seg_id}.png",
        5
    )
    all_results.append(results)

jax3dp3.setup_visualizer()
# Viz results
idx = 0
(score, overlap, pose, obj_idx, image, image_unmasked, prob) = all_results[idx][0]

jax3dp3.clear()
jax3dp3.show_cloud("actual", t3d.point_cloud_image_to_points(point_cloud_image_above_table))
jax3dp3.show_cloud("pred", t3d.apply_transform(online_state.meshes[obj_idx].vertices,pose), color=np.array([1.0, 0.0, 0.0]))




