
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

# filename = "panda_dataset/scene_4.pkl"
# filename = "panda_dataset_2/utensils.pkl"
# full_filename = "blue.pkl"
full_filename = "1674620488.514845.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "knife_spoon.pkl"
full_filename = "utensils.pkl"
full_filename = "new_utencils.pkl"
full_filename = "utensils.pkl"
full_filename = "nishad_0.pkl"
full_filename = "shape_acquisition.pkl"



full_filename = "nishad_1.pkl"
full_filename = "knife_spoon.pkl"



full_filename = "demo2_nolight.pkl"

full_filename = "strawberry_error.pkl"

full_filename = "knife_sim.pkl"

file = open(full_filename,'rb')
camera_images = pickle.load(file)["camera_images"]
file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

# jax3dp3.setup_visualizer()

orig_h, orig_w = observations[0].camera_params[:2]

state = jax3dp3.OnlineJax3DP3()
state.set_camera_parameters(observations[0].camera_params, scaling_factor=0.3)

observation = observations[-1]
point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
state.infer_table_plane(point_cloud_image, observation.camera_pose)

state.start_renderer()

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
for (name, path) in zip(model_names, model_paths):
    state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )

obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
segmentation_image  = state.segment_scene(
    observation.rgb, obs_point_cloud_image, observation.camera_pose,
    FLOOR_THRESHOLD=0.0,
    TOO_CLOSE_THRESHOLD=0.2,
    FAR_AWAY_THRESHOLD=0.8,
    viz_filename="dashboard.png"
)

timestep = 1
unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]

state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.05, 0.04, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(5,5,21),(5,5,21),(15, 15, 1), (5,5,21)],
)

unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]

timestep = 1
outlier_volume = 1**3

r = 0.005
outlier_prob = 0.05

# possible_rs = jnp.array([0.01, 0.008, 0.006, 0.004])
# possible_outlier_prob = jnp.array([0.1, 0.05])

for seg_id in segmetation_ids:
    obs_image_masked, obs_image_complement = state.get_image_masked_and_complement(
        obs_point_cloud_image, segmentation_image, seg_id
    )
    contact_init = state.infer_initial_contact_parameters(
        obs_image_masked, observation.camera_pose
    )

    object_ids_to_estimate = jnp.arange(len(model_paths))
    latent_hypotheses = []
    for obj_idx in object_ids_to_estimate:
        latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

    hypotheses_over_time = jax3dp3.c2f.c2f_contact_parameters(
        latent_hypotheses,
        state.contact_param_sched,
        state.face_param_sched,
        r,
        jnp.linalg.inv(observation.camera_pose) @ state.table_surface_plane_pose,
        obs_point_cloud_image,
        obs_image_complement,
        outlier_prob,
        outlier_volume,
        state.model_box_dims
    )

    (h,w,fx,fy,cx,cy, near, far) = state.camera_params


    obs_data_images = [
            jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2], max=far), orig_h, orig_w),
            jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_masked[:,:,2], max=far), orig_h, orig_w),
    ]
    rgb_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(observation.rgb, 255.0), orig_h, orig_w)

    viz_panels = []
    for idx in range(len(hypotheses_over_time[0])):
        labels = ["Obs", "Masked"]

        viz_images = []
        for hypotheses in hypotheses_over_time[1:]:
            (score, obj_idx, _, pose) = hypotheses[idx]
            depth = jax3dp3.render_single_object(pose, obj_idx)
            depth_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(depth[:,:,2], max=far), orig_h, orig_w)
            viz_images.append(
                jax3dp3.viz.overlay_image(
                    rgb_viz,
                    depth_viz
                )
            )

        viz_panels.append(
            jax3dp3.viz.multi_panel(
                obs_data_images + viz_images, labels, title="{:s}".format(state.mesh_names[obj_idx])
            )
        )

    scores = jnp.array( [i[0] for i in hypotheses_over_time[-1]])
    order = jnp.argsort(-scores)
    normalized_scores = jax3dp3.utils.normalize_log_scores(scores)
    print(normalized_scores)

    viz_panels_sorted = []
    scores_string = []
    for i in order:
        viz_panels_sorted.append(
            viz_panels[i]
        )
        scores_string.append(
            "{:0.2f}".format(normalized_scores[i])
        )
    jax3dp3.viz.multi_panel(
        [jax3dp3.viz.vstack_images(viz_panels_sorted)],
        title="Distribution = {:s}".format(
            ", ".join(scores_string)
        )
    ).save(f"seg_id_{seg_id}.png")

    # from IPython import embed; embed()



from IPython import embed; embed()



