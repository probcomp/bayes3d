import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3

jax3dp3.setup_visualizer()


# filename = "panda_dataset/scene_4.pkl"
filename = "panda_dataset_2/utensils.pkl"
print(f"Processing scene {filename}...")
file = open(os.path.join(jax3dp3.utils.get_assets_dir(), filename),'rb')
all_data = pickle.load(file)
file.close()

K = jnp.array([[606.92871094, 0.    , 415.18270874],
    [ 0.    , 606.51989746, 480 - 258.89492798],
    [ 0.    , 0.    , 1.    ]])
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]

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
near = 0.001
far = 4.99


def pre_process_data(data):
    rgb = data["rgb"]
    depth_original = data["depth"] / 1000.0
    translation = jnp.array(data["extrinsics"][0])
    R = t3d.xyzw_to_rotation_matrix(data["extrinsics"][1])
    cam_pose = (
        t3d.transform_from_rot_and_pos(R, translation) 
        @ 
        # t3d.inverse_pose(gripper_pose_to_cam_pose)
        gripper_pose_to_cam_pose
    )
    return rgb, depth_original, cam_pose

rgb, depth_original, camera_pose = pre_process_data(all_data[0])
orig_h = depth_original.shape[0]
orig_w = depth_original.shape[1]

online_state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.1, 0.07, 0.04, 0.02],
    grid_params=[(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20)],
    likelihood_r_sched = [0.2, 0.15, 0.1, 0.04, 0.02]
)

online_state.set_camera_params(
    orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, near, far,
    scaling_factor=0.3
)


point_cloud_image = online_state.process_depth_to_point_cloud_image(np.array(depth_original))
online_state.infer_table_plane(point_cloud_image, camera_pose)

(h,w,fx,fy,cx,cy, near, far) = online_state.camera_params
outlier_prob, outlier_volume = 0.1, 10**3

rgb_original = np.array(rgb)
point_cloud_image = online_state.process_depth_to_point_cloud_image(
    np.array(depth_original))

point_cloud_image_above_table, segmentation_image  = online_state.segment_scene(
    rgb_original, point_cloud_image, camera_pose, f"dashboard.png"
)


from IPython import embed; embed()



online_state = jax3dp3.Jax3DP3Perception()




pcs = []
poses = []

for t in range(4):
    rgb = all_data[t]["rgb"]
    jax3dp3.viz.save_rgb_image(rgb, 255.0, f"imgs/{t}.png")
    depth_original = all_data[t]["depth"] / 1000.0


    gt_image_full = t3d.depth_to_point_cloud_image(depth_original, fx, fy, cx, cy)
    point_cloud = t3d.point_cloud_image_to_points(gt_image_full)

    translation = jnp.array(all_data[t]["extrinsics"][0])
    R = t3d.xyzw_to_rotation_matrix(all_data[t]["extrinsics"][1])
    cam_pose = (
        t3d.transform_from_rot_and_pos(R, translation) 
        @ 
        # t3d.inverse_pose(gripper_pose_to_cam_pose)
        gripper_pose_to_cam_pose
    )


    pcs.append(point_cloud)
    poses.append(cam_pose)


all_clouds = []
for (pc, pose) in zip(pcs, poses):
    all_clouds.append(
        t3d.apply_transform(pc, pose)
    )

jax3dp3.clear()

jax3dp3.show_cloud("a", all_clouds[0])
jax3dp3.show_pose("a2", poses[0])
jax3dp3.show_pose("a3", poses[1])
jax3dp3.show_pose("a4", poses[2])
jax3dp3.show_pose("a5", poses[3])

jax3dp3.show_cloud("a4", all_clouds[1])

jax3dp3.show_cloud("a4", t3d.apply_transform(pcs[1], poses[1]))


jax3dp3.show_pose("a5", poses[2])
jax3dp3.show_pose("a6", poses[3])

from IPython import embed; embed()

jax3dp3.show_cloud("a", pcs[0])
jax3dp3.show_cloud("a", pcs[0])





Ks = [all_data[t]["intrinsics"][0] for t in range(4)]



rgb_original = data["rgb"]
jax3dp3.viz.save_rgb_image(rgb_original, 255.0, "imgs/0.png")


from IPython import embed; embed()



rgb_original = data["rgb"]
depth_original = data["depth"] / 1000.0
K = data["intrinsics"][0]
orig_h,orig_w = depth_original.shape
K = jnp.array([[606.92871094, 0.    , 415.18270874],
    [ 0.    , 606.51989746, 258.89492798],
    [ 0.    , 0.    , 1.    ]])
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
near = 0.001
far = 5.0