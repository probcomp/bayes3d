import pickle as pkl
import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import os
import trimesh
import jax.numpy as jnp

with open("data.pkl","rb") as f:
    d = pkl.load(f)

depth = d["depth_image"]
floor_plane = d["floor_plane"]
h,w,fx,fy,cx,cy,near,far = (
    d["camera_intrinsics"]["height"],
    d["camera_intrinsics"]["width"],
    d["camera_intrinsics"]["fx"],
    d["camera_intrinsics"]["fy"],
    d["camera_intrinsics"]["cx"],
    d["camera_intrinsics"]["cy"],
    d["camera_intrinsics"]["near"],
    d["camera_intrinsics"]["far"],
)

state = j.OnlineJax3DP3()
state.start_renderer((h,w,fx,fy,cx,cy,near,far),scaling_factor=0.3)
obs_point_cloud_image = state.process_depth_to_point_cloud_image(depth)


j.setup_visualizer()
j.show_cloud("1", t3d.point_cloud_image_to_points(obs_point_cloud_image))

mesh = trimesh.load(os.path.join(j.utils.get_assets_dir(), "sample_objs/sphere.obj"))
mesh.vertices = mesh.vertices * 0.35
j.show_cloud("2", mesh.vertices)

state.add_trimesh(mesh,"ball")


surface_plane = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2),
    jnp.array([0.0, floor_plane, 5.0])
)
j.show_cloud("1", t3d.apply_transform(t3d.point_cloud_image_to_points(obs_point_cloud_image), jnp.eye(4)))
j.show_cloud("1", t3d.apply_transform(t3d.point_cloud_image_to_points(obs_point_cloud_image), t3d.inverse_pose(surface_plane) ))

table_dims = jnp.array([20.0, 20.0])
grid_params = (50,50,1)
contact_param_sweep, face_param_sweep = j.scene_graph.enumerate_contact_and_face_parameters(
    -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
    *grid_params,
    jnp.arange(1)
)

segmentation_image = obs_point_cloud_image[:,:,0]
good_poses, pose_proposals, ranked_high_value_seg_ids = j.c2f.c2f_occluded_object_pose_distribution(
    0,
    segmentation_image,
    contact_param_sweep,
    face_param_sweep,
    0.01,
    surface_plane,
    obs_point_cloud_image,
    0.01,
    1.0,
    state.model_box_dims,
    state.camera_params,
)
print(good_poses.shape)

far_plane = 10.0
depth_viz = j.viz.get_depth_image(obs_point_cloud_image[:,:,2],max=far_plane)
depth_viz.save("depth.png")

good_poses_image = j.render_multiobject(good_poses, [0 for _ in range(good_poses.shape[0])])
posterior = j.viz.get_depth_image(good_poses_image[:,:,2], max=far)

j.viz.multi_panel(
    [
        depth_viz,
        posterior,
        j.viz.overlay_image(depth_viz, posterior, alpha=0.7)
    ],
    labels=["Observed Depth", "Posterior", "Overlay"],
    label_fontsize=15
).save("overlay.png")


occlusion_viz.save("occlusion.png")

j.clear()
j.show_cloud("1", t3d.point_cloud_image_to_points(obs_point_cloud_image))
j.show_cloud("3", t3d.apply_transform(mesh.vertices,good_poses[0]), color=j.RED)
