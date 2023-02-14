import jax3dp3
import trimesh
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import os

observation, gt_ids, gt_poses, masks = jax3dp3.ycb_loader.get_test_img(
    '53', '1', "/home/nishadgothoskar/data/bop/ycbv"
)
jax3dp3.viz.get_rgb_image(observation.rgb, 255.0).save("rgb.png")

state = jax3dp3.OnlineJax3DP3()
state.start_renderer(observation.camera_params)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
model_names = jax3dp3.ycb_loader.MODEL_NAMES
meshes = [trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply")) for idx in range(21)]
for (mesh, mesh_name) in zip(meshes, model_names):
    state.add_trimesh(mesh, mesh_name, mesh_scaling_factor=1.0/1000.0)

state.table_surface_plane_pose  = jnp.eye(4)

img = jax3dp3.render_multiobject(gt_poses, gt_ids)
jax3dp3.viz.get_depth_image(img[:,:,2],max=2.0).save("reconstruction.png")


state.set_coarse_to_fine_schedules(
    grid_widths=[0.1, 0.01, 0.04, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(7, 7 ,21),(7, 7 ,21),(15, 15, 1), (7, 7 ,21)],
)



hypotheses_over_time = state.inference_for_segment(
    state.process_depth_to_point_cloud_image(observation.depth),
    state.process_segmentation_mask(masks[2]*1.0),
    1.0,
    gt_ids,
    observation.camera_pose, 
    r=0.01,
    outlier_prob=0.1
)
scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
print(jax3dp3.utils.normalize_log_scores(scores))


(h,w,fx,fy,cx,cy, near, far) = state.camera_params
orig_h, orig_w = observation.rgb.shape[:2]
images = []
rgb_viz = jax3dp3.viz.get_rgb_image(observation.rgb)
for hyp in hypotheses_over_time[-1]:
    (score, obj_idx, _, pose) = hyp
    depth = jax3dp3.render_single_object(pose, obj_idx)
    depth_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(depth[:,:,2], max=1.0),orig_h, orig_w)
    images.append(jax3dp3.viz.multi_panel([jax3dp3.viz.overlay_image(rgb_viz, depth_viz)],title="{:0.4f}".format(score)))


jax3dp3.viz.multi_panel(images).save("predictions.png")








state = jax3dp3.OnlineJax3DP3()
