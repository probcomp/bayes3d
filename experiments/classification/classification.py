
import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
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

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")


test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/lego_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/spoon_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]

images = [j.RGBD.construct_from_camera_image(c) for c in camera_images]
intrinsics = j.camera.scale_camera_parameters(images[0].intrinsics, 0.3)
renderer = j.Renderer(intrinsics)

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
for (path, n) in zip(model_paths, mesh_names):
    renderer.add_mesh(j.mesh.center_mesh(trimesh.load(path)),n)


depth_scaled =  j.utils.resize(images[0].depth, intrinsics.height, intrinsics.width)
obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)
import jax3dp3.segment_scene
segmentation_image, mask, viz = jax3dp3.segment_scene.segment_scene(
    images[0].rgb,
    obs_point_cloud_image,
    intrinsics
)
viz.save("viz.png")

from IPython import embed; embed()

segmentation_id = 0
depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, segmentation_image, segmentation_id, intrinsics)
j.get_depth_image(depth_masked, max=intrinsics.far).save("masked.png")
j.get_depth_image(depth_complement, max=intrinsics.far).save("complement.png")
obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)

table_plane, table_dims = j.utils.infer_table_plane(obs_point_cloud_image, images[0].camera_pose, intrinsics)
# j.setup_visualizer()
# j.show_cloud("1", 
#     t3d.apply_transform(
#         obs_point_cloud_image.reshape(-1,3),
#         t3d.inverse_pose(table_plane) @ images[0].camera_pose
#     ) 
# )


sched = j.c2f.make_schedules(
    grid_widths=[0.05, 0.03, 0.02, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(7,7,21),(7,7,21),(15, 15, 1), (7,7,21)],
)

r_sweep = jnp.array([0.02])
outlier_prob=0.1
outlier_volume=1.0

model_box_dims = jnp.array([j.utils.aabb(m.vertices)[0] for m in renderer.meshes])
hypotheses_over_time = j.c2f.c2f_contact_parameters(
    renderer,
    obs_point_cloud_image,
    obs_point_cloud_image_masked,
    obs_point_cloud_image_complement,
    sched,
    t3d.inverse_pose(images[0].camera_pose) @  table_plane,
    r_sweep,
    outlier_prob,
    outlier_volume,
    model_box_dims
)


scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
normalized_scores = j.utils.normalize_log_scores(scores)
order = np.argsort(-np.array(scores))


orig_h, orig_w = images[0].rgb.shape[:2]
rgb_viz = j.get_rgb_image(images[0].rgb)
mask = j.utils.resize((segmentation_image == segmentation_id)* 1.0, orig_h,orig_w)[...,None]
rgb_masked_viz = j.viz.get_rgb_image(
    images[0].rgb * mask
)
viz_images = [
    rgb_viz,
    j.viz.overlay_image(rgb_viz, rgb_masked_viz, alpha=0.6)
]
top = j.viz.multi_panel(
    viz_images, 
    labels=["RGB Input", "Segment to Classify"],
    label_fontsize=50    
)

viz_images = []
labels = []
for i in order:
    (score, obj_idx, _, pose) = hypotheses_over_time[-1][i]
    depth = renderer.render_single_object(pose, obj_idx)
    depth_viz = j.viz.resize_image(j.viz.get_depth_image(depth[:,:,2], max=1.0), images[0].rgb.shape[0], images[0].rgb.shape[1])
    viz_images.append(
        j.viz.multi_panel(
            [j.viz.overlay_image(rgb_viz, depth_viz)],
            labels=[
                    "{:s} - {:0.2f}".format(
                    renderer.mesh_names[obj_idx],
                    normalized_scores[i]
                )
            ],
            label_fontsize=50
        )
    )
final_viz = j.viz.vstack_images(
    [top, *viz_images], border= 20
)
final_viz.save("final.png")






exact_match_score = j.threedp3_likelihood_parallel_jit(
    obs_point_cloud_image, jnp.array([obs_point_cloud_image]), r_sweep[0], outlier_prob, outlier_volume
)[0]
final_scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
known_object_scores = (jnp.array(final_scores) - exact_match_score) / ((segmentation_image == segmentation_id).sum()) * 1000.0




from IPython import embed; embed()


from IPython import embed; embed()


viz.save("depth.png")




from IPython import embed; embed()

