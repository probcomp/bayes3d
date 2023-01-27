
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
full_filename = "strawberry_error.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "knife_spoon.pkl"
full_filename = "utensils.pkl"
full_filename = "new_utencils.pkl"
full_filename = "utensils.pkl"
full_filename = "nishad_0.pkl"
file = open(full_filename,'rb')
camera_image = pickle.load(file)["camera_images"][0]
file.close()

depth = np.array(camera_image.depthPixels) 
camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
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

point_cloud_image = online_state.process_depth_to_point_cloud_image(depth)
jax3dp3.viz.save_depth_image(point_cloud_image[:,:,2],"depth.png",max=far)

online_state.infer_table_plane(point_cloud_image, camera_pose)


point_cloud_image_above_table, segmentation_image  = online_state.segment_scene(
    rgb_original, point_cloud_image, camera_pose, "dashboard.png",
    FLOOR_THRESHOLD=-0.01,
    TOO_CLOSE_THRESHOLD=0.2,
    FAR_AWAY_THRESHOLD=0.8,
)


from IPython import embed; embed()

jax3dp3.setup_visualizer()
jax3dp3.show_cloud("c1",t3d.apply_transform(
    t3d.point_cloud_image_to_points(point, 
    camera_pose
)))



# jax3dp3.setup_visualizer()

# jax3dp3.show_cloud("c1",t3d.apply_transform(t3d.point_cloud_image_to_points(point_cloud_image_above_table), camera_pose))



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





