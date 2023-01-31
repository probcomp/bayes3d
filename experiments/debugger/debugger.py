
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
full_filename = "knife_spoon.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "knife_spoon.pkl"
full_filename = "utensils.pkl"
full_filename = "new_utencils.pkl"
full_filename = "utensils.pkl"
full_filename = "nishad_0.pkl"
full_filename = "nishad_1.pkl"
full_filename = "shape_acquisition.pkl"

full_filename = "strawberry_error.pkl"

full_filename = "demo2_nolight.pkl"
full_filename = "knife_sim.pkl"

file = open(full_filename,'rb')
camera_images = pickle.load(file)["camera_images"]
file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=5.0) for img in camera_images]
print('len(observations):');print(len(observations))

# jax3dp3.setup_visualizer()

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

outlier_prob, outlier_volume = 0.01, 10**3
timestep = 1
unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]


perception_state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.1, 0.07, 0.04, 0.02, 0.02],
    angle_widths=[jnp.pi, jnp.pi, jnp.pi, jnp.pi/2, jnp.pi/4],
    grid_params=[(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20)],
    likelihood_r_sched = [0.06, 0.05, 0.04, 0.03, 0.02]
)


unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]



seg_id = segmetation_ids[1]

contact_param_sched = perception_state.contact_param_sched
likelihood_r_sched = perception_state.likelihood_r_sched
face_param_sched = perception_state.face_param_sched
model_box_dims = perception_state.model_box_dims

obs_image_masked, obs_image_complement = perception_state.get_image_masked_and_complement(
    obs_point_cloud_image, segmentation_image, seg_id
)
contact_init = perception_state.infer_initial_contact_parameters(
    obs_image_masked, observation.camera_pose
)
results = perception_state.run_detection(
    observation.rgb,
    obs_image_masked,
    obs_image_complement,
    contact_init,
    observation.camera_pose,
    outlier_prob,
    outlier_volume,
    0.06,
    0.02,
    f"best.png",
    5
)


timestep = 1
outlier_prob, outlier_volume = 0.01, 10**3

latent_hypotheses = []
for obj_idx in range(model_box_dims.shape[0]):
    latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

latent_hypotheses_over_time = []
for c2f_iter in range(len(likelihood_r_sched)):
    new_latent_hypotheses = []

    r = likelihood_r_sched[c2f_iter]
    contact_param_sweep_delta, face_param_sweep = contact_param_sched[c2f_iter], face_param_sched[c2f_iter]
    print(r)
    for hypothesis in latent_hypotheses:
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
        keep_mask = jnp.logical_or(
            (rendered_images_unmasked[:,:,:,2] <= obs_image_complement[None, :,:, 2]),
            (obs_image_complement[:,:,2] == 0)[None, ...]
        )[...,None]
        rendered_images = keep_mask * rendered_images_unmasked # + (1.0 - keep_mask) * obs_image_complement
        # rendered_images = jax3dp3.renderer.get_complement_masked_images(rendered_images_unmasked, obs_image_complement)
        # weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume)
        weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_image_masked, rendered_images, r, outlier_prob, outlier_volume)
        best_idx = weights.argmax()
        print(weights[best_idx])

        new_latent_hypotheses.append(
            (weights[best_idx], obj_idx, new_contact_param_sweep[best_idx],  rendered_images[best_idx], rendered_images_unmasked[best_idx], keep_mask[best_idx])
        )

    scores = []
    for hypothesis in new_latent_hypotheses:
        scores.append(hypothesis[0])

    new_latent_hypotheses_sorted = []
    for idx in np.argsort(-np.array(scores)):
        new_latent_hypotheses_sorted.append(new_latent_hypotheses[idx])
    
    latent_hypotheses_over_time.append(new_latent_hypotheses_sorted)
    latent_hypotheses = new_latent_hypotheses_sorted


(h,w,fx,fy,cx,cy, near, far) = perception_state.camera_params

(score, obj_idx, cp, rendered_image, rendered_image_unmasked, keep_mask) = latent_hypotheses[1]
jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_masked[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_complement[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(rendered_image_unmasked[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(rendered_image[:,:,2], max=far),h,w),
    ],
    labels=[ "Obs","Masked Obs", "Complement","Unmasked", "Final"],
).save("best.png")


from IPython import embed; embed()





all_results = []
for seg_id in segmetation_ids:
    image_masked, image_complement = perception_state.get_image_masked_and_complement(
        point_cloud_image, segmentation_image, seg_id
    )
    contact_init = perception_state.infer_initial_contact_parameters(
        image_masked, observation.camera_pose
    )

    results = perception_state.run_detection(
        observation.rgb,
        image_masked,
        image_complement,
        contact_init,
        observation.camera_pose,
        outlier_prob,
        outlier_volume,
        0.06,
        0.03,
        f"out_{timestep}_seg_id_{seg_id}.png",
        5
    )
    all_results.append(results)

from IPython import embed; embed()


(h,w,fx,fy,cx,cy, near, far) = perception_state.camera_params

jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(observation.rgb, 255.0),h,w),
        jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2],  max=far),
        jax3dp3.viz.get_depth_image(above_table_image[:,:,2],  max=far),
        jax3dp3.viz.get_depth_image(segmentation_image + 1, max=segmentation_image.max() + 1),
    ],
    labels=["RGB", "Depth", "Above Table", "Segmentation. {:d}".format(int(segmentation_image.max() + 1))],
).save("dashboard.png")




unique =  np.unique(segmentation_image)
segmetation_ids = unique[unique != -1]



seg_id = segmetation_ids[1]

contact_param_sched = perception_state.contact_param_sched
likelihood_r_sched = perception_state.likelihood_r_sched
face_param_sched = perception_state.face_param_sched
model_box_dims = perception_state.model_box_dims

obs_image_masked, obs_image_complement = perception_state.get_image_masked_and_complement(
    obs_point_cloud_image, segmentation_image, seg_id
)
contact_init = perception_state.infer_initial_contact_parameters(
    obs_image_masked, observation.camera_pose
)





timestep = 1
outlier_prob, outlier_volume = 0.01, 10**3

latent_hypotheses = []
for obj_idx in range(model_box_dims.shape[0]):
    latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

latent_hypotheses_over_time = []
for c2f_iter in range(len(likelihood_r_sched)):
    new_latent_hypotheses = []

    r = likelihood_r_sched[c2f_iter]
    contact_param_sweep_delta, face_param_sweep = contact_param_sched[c2f_iter], face_param_sched[c2f_iter]
    print(r)
    for hypothesis in latent_hypotheses:
        obj_idx = hypothesis[1]
        contact_params = hypothesis[2]
        new_contact_param_sweep = contact_param_sweep_delta + contact_param_sweep_delta  # shift center 

        pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
            new_contact_param_sweep,
            face_param_sweep,
            model_box_dims[obj_idx],
            jnp.linalg.inv(observation.camera_pose) @ perception_state.table_surface_plane_pose
        )

        # get best pose proposal
        rendered_images_unmasked = jax3dp3.render_parallel(pose_proposals, obj_idx)[...,:3]
        keep_mask = (
            (rendered_images_unmasked[:,:,:,2] < obs_image_complement[None, :,:, 2]) *
            (obs_image_complement[:,:,2] > 0)[None, ...]
        )[...,None]
        rendered_images = keep_mask * rendered_images_unmasked # + (1.0 - keep_mask) * obs_image_complement
        # weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_point_cloud_image, rendered_images, r, outlier_prob, outlier_volume)
        weights = jax3dp3.threedp3_likelihood_parallel_jit(obs_image_masked, rendered_images, r, outlier_prob, outlier_volume)
        best_idx = weights.argmax()
        print(weights[best_idx])

        new_latent_hypotheses.append(
            (weights[best_idx], obj_idx, new_contact_param_sweep[best_idx],  rendered_images[best_idx], rendered_images_unmasked[best_idx], keep_mask[best_idx])
        )

    scores = []
    for hypothesis in new_latent_hypotheses:
        scores.append(hypothesis[0])

    new_latent_hypotheses_sorted = []
    for idx in np.argsort(-np.array(scores)):
        new_latent_hypotheses_sorted.append(new_latent_hypotheses[idx])
    
    latent_hypotheses_over_time.append(new_latent_hypotheses_sorted)
    latent_hypotheses = new_latent_hypotheses_sorted


(score, obj_idx, cp, rendered_image, rendered_image_unmasked, keep_mask) = latent_hypotheses[0]
jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_masked[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_image_complement[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(rendered_image_unmasked[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(keep_mask[:,:,0], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(rendered_image[:,:,2], max=far),h,w),
    ],
    labels=[ "Obs","Masked Obs", "Complement","Unmasked", "Mask","Final"],
).save("best.png")

from IPython import embed; embed()

image = latent_hypotheses[0][-1]
jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_image[:,:,2], max=far),h,w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(image[:,:,2], max=far),h,w),
    ],
    labels=["Observation", "RGB"],
).save("best.png")

from IPython import embed; embed()


contact_param_sweep = contact_param_sweep_delta + c  # shift center 

# get pose proposals in cam frame
pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
    contact_param_sweep,
    face_param_sweep,
    model_box_dims[obj_idx],
    contact_plane_pose
)  


all_results = []
for seg_id in segmetation_ids:


    results = perception_state.run_detection(
        observation.rgb,
        image_masked,
        image_complement,
        contact_init,
        observation.camera_pose,
        outlier_prob,
        outlier_volume,
        0.06,
        0.03,
        f"out_{timestep}_seg_id_{seg_id}.png",
        5
    )
    all_results.append(results)


from IPython import embed; embed()




# # Viz results
# idx = 0
# (score, overlap, pose, obj_idx, image, rendered_image_unmasked, prob) = all_results[idx][0]


# seg_id = segmetation_ids[idx]
# image_masked, image_complement = perception_state.get_image_masked_and_complement(
#     point_cloud_image, segmentation_image, seg_id
# )
# rendered_image_unmasked = jax3dp3.render_single_object(pose, obj_idx)
# rendered_image = jax3dp3.renderer.get_complement_masked_image(rendered_image_unmasked, image_complement)
# weight = jax3dp3.threedp3_likelihood_jit(image_masked, rendered_image, 0.07, outlier_prob, outlier_volume)

# jax3dp3.viz.multi_panel(
#     [
#         jax3dp3.viz.get_depth_image(rendered_image[:,:,2], max=5.0),
#         jax3dp3.viz.get_depth_image(image_masked[:,:,2], max=5.0),
#     ]
# ).save("out.png")

# rs_to_check = jnp.linspace(0.005, 0.07, 20)
# outlier_prob, outlier_volume = 0.1, 10**3
# weights = []
# for r in rs_to_check:
#     weight = jax3dp3.threedp3_likelihood_jit(image_masked, rendered_image, r, outlier_prob, outlier_volume)
#     print(f"r: {r} weight: {weight}")
#     weights.append(weight)
# print(rs_to_check[jnp.argmax(jnp.array(weights))])

# outlier_prob_to_check = jnp.linspace(0.001, 0.2, 50)
# weights = []
# for op in outlier_prob_to_check:
#     weight = jax3dp3.threedp3_likelihood(image_masked, rendered_image, 0.06, op, outlier_volume)
#     print(f"outlier_prob: {op} weight: {weight}")
#     weights.append(weight)
# print(outlier_prob_to_check[jnp.argmax(jnp.array(weights))])

# # jax3dp3.setup_visualizer()
# jax3dp3.clear()
# jax3dp3.show_cloud("actual", t3d.apply_transform(t3d.point_cloud_image_to_points(point_cloud_image_above_table), observation.camera_pose))
# jax3dp3.show_cloud("pred", t3d.apply_transform(perception_state.meshes[obj_idx].vertices, observation.camera_pose @ pose), color=np.array([1.0, 0.0, 0.0]))

# top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
# model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "banana"]
# model_paths = [
#     os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
#     os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
#     os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
#     os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
#     os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
#     os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
# ]


# for (name, path) in zip(model_names, model_paths):
#     perception_state.add_trimesh(
#         trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
#     )