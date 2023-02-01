
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

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=1.0) for img in camera_images]
print('len(observations):');print(len(observations))

# jax3dp3.setup_visualizer()

orig_h, orig_w = observations[0].camera_params[:2]

perception_state = jax3dp3.OnlineJax3DP3()
perception_state.set_camera_parameters(observations[0].camera_params, scaling_factor=0.2)

observation = observations[-1]
point_cloud_image = perception_state.process_depth_to_point_cloud_image(observation.depth)
perception_state.infer_table_plane(point_cloud_image, observation.camera_pose)

perception_state.start_renderer()

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
    perception_state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )

obs_point_cloud_image = perception_state.process_depth_to_point_cloud_image(observation.depth)
segmentation_image  = perception_state.segment_scene(
    observation.rgb, obs_point_cloud_image, observation.camera_pose,
    FLOOR_THRESHOLD=0.005,
    TOO_CLOSE_THRESHOLD=0.2,
    FAR_AWAY_THRESHOLD=0.8,
    viz_filename="dashboard.png"
)

timestep = 1
unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]


# perception_state.set_coarse_to_fine_schedules(
#     grid_widths=[0.08, 0.05, 0.04, 0.02, 0.02],
#     angle_widths=[jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi],
#     grid_params=[(10,10, 31),(10,10, 31),(10,10, 31),(10,10, 31),(10,10, 31)],
#     likelihood_r_sched = [0.08, 0.05, 0.02, 0.01, 0.005]
# )


perception_state.set_coarse_to_fine_schedules(
    grid_widths=[0.08, 0.05, 0.04, 0.03, 0.02, 0.00001],
    angle_widths=[jnp.pi, jnp.pi, jnp.pi, jnp.pi/2, 0.001, jnp.pi],
    grid_params=[(3,3,27),(3,3,27),(5,5,27),(5,5,17), (10, 10, 1), (1,1,360)],
    likelihood_r_sched = [0.08, 0.05, 0.02, 0.01, 0.01, 0.01]
)

contact_param_sched = perception_state.contact_param_sched
likelihood_r_sched = perception_state.likelihood_r_sched
face_param_sched = perception_state.face_param_sched
model_box_dims = perception_state.model_box_dims


unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]

timestep = 1
outlier_prob, outlier_volume = 0.1, 10**3



for seg_id in segmetation_ids:
    obs_image_masked, obs_image_complement = perception_state.get_image_masked_and_complement(
        obs_point_cloud_image, segmentation_image, seg_id
    )
    contact_init = perception_state.infer_initial_contact_parameters(
        obs_image_masked, observation.camera_pose
    )

    object_ids_to_estimate = jnp.arange(len(model_paths))
    latent_hypotheses = []
    for obj_idx in object_ids_to_estimate:
        latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

    latent_hypotheses_over_time = [latent_hypotheses]
    for c2f_iter in range(len(likelihood_r_sched)):
        new_latent_hypotheses = []

        r = likelihood_r_sched[c2f_iter]
        contact_param_sweep_delta, face_param_sweep = contact_param_sched[c2f_iter], face_param_sched[c2f_iter]
        print(r)
        for hypothesis in latent_hypotheses:
            old_score = hypothesis[0]
            obj_idx = hypothesis[1]
            contact_params = hypothesis[2]
            new_contact_param_sweep = contact_params + contact_param_sweep_delta  # shift center 

            pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
                new_contact_param_sweep,
                face_param_sweep,
                model_box_dims[obj_idx],
                jnp.linalg.inv(observation.camera_pose) @ perception_state.table_surface_plane_pose
            )

            # get best pose proposal
            rendered_images_unmasked = jax3dp3.render_parallel(pose_proposals, obj_idx)[...,:3]
            # keep_mask = jnp.logical_or(
            #     (rendered_images_unmasked[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * rendered_images_unmasked[:,:,:,2] > 0.0,
            #     (obs_image_complement[:,:,2] != 0)[None, ...]
            # )[...,None]
            keep_mask = jnp.logical_or(
                (rendered_images_unmasked[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * 
                rendered_images_unmasked[:,:,:,2] > 0.0
                ,
                (obs_image_complement[:,:,2] == 0)[None, ...]
            )[...,None]
            rendered_images = keep_mask * rendered_images_unmasked + (1.0 - keep_mask) * obs_image_complement
            # rendered_images = jax3dp3.renderer.get_complement_masked_images(rendered_images_unmasked, obs_image_complement)
            # weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume)
            weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume)
            best_idx = weights.argmax()
            new_score = weights[best_idx]
            print(new_score, new_score > old_score)

            new_latent_hypotheses.append(
                (weights[best_idx], obj_idx, new_contact_param_sweep[best_idx],  rendered_images[best_idx], rendered_images_unmasked[best_idx], keep_mask[best_idx])
            )

        latent_hypotheses_over_time.append(new_latent_hypotheses)
        latent_hypotheses = new_latent_hypotheses

    (h,w,fx,fy,cx,cy, near, far) = perception_state.camera_params

    viz_panels = []
    for (i,obj_idx) in enumerate(object_ids_to_estimate):
        viz_images = [
                jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2], max=far), orig_h, orig_w),
                jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_masked[:,:,2], max=far), orig_h, orig_w),
        ]

        labels = ["Obs", "Masked"]

        for hypotheses in latent_hypotheses_over_time[1:]:
            (score, obj_idx, cp, rendered_image, rendered_image_unmasked, keep_mask) = hypotheses[i]
            viz_images.append(
                jax3dp3.viz.overlay_image(
                        jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(observation.rgb, 255.0), orig_h, orig_w),
                        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(rendered_image_unmasked[:,:,2], max=far), orig_h, orig_w)
                )
            )
            labels += [f"{score}"]

        viz_panels.append(jax3dp3.viz.multi_panel(
            viz_images, labels, title="{:s}".format(perception_state.mesh_names[obj_idx])
        )
        )


    scores = []
    for hypothesis in latent_hypotheses_over_time[-1]:
        weight = jax3dp3.threedp3_likelihood_jit(
            obs_point_cloud_image, hypothesis[3],
            0.02, outlier_prob, outlier_volume
        )
        scores.append(weight)
    scores = jnp.array(scores)
    scores = jax3dp3.utils.normalize_log_scores(scores)

    viz_panels = [
        viz_panels[i]
        for i in jnp.argsort(-scores)
    ]
    scores_string = []
    for i in jnp.argsort(-scores):
        scores_string.append(
            "{:0.2f}".format(scores[i])
        )
    jax3dp3.viz.multi_panel(
        [jax3dp3.viz.vstack_images(viz_panels)],
        title="Distribution = {:s}".format(
            ", ".join(scores_string)
        )
    ).save(f"seg_id_{seg_id}.png")



from IPython import embed; embed()



