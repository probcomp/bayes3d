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

far=10.0
original_intrinsics = j.Intrinsics(h,w,fx,fy,cx,cy,near,far)
intrinsics = j.camera.scale_camera_parameters(original_intrinsics, 0.3)
renderer = j.Renderer(intrinsics)

mesh = trimesh.load(os.path.join(j.utils.get_assets_dir(), "sample_objs/sphere.obj"))
mesh.vertices = mesh.vertices * 0.35
renderer.add_mesh(mesh)

depth_scaled =  j.utils.resize(depth, intrinsics.height, intrinsics.width)
segmentation_image  = 1.0 * (depth_scaled > intrinsics.near) * (depth_scaled < intrinsics.far)
depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, segmentation_image, 1.0, intrinsics)
obs_point_cloud_image = j.t3d.unproject_depth(depth_scaled, intrinsics)

from IPython import embed; embed()

contact_plane_pose_in_cam_frame = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2),
    jnp.array([0.0, floor_plane, 5.0])
)


table_dims = jnp.array([20.0, 20.0])
grid_params = (50,50,1)
contact_param_sweep, face_param_sweep = j.scene_graph.enumerate_contact_and_face_parameters(
    -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
    *grid_params,
    jnp.arange(1)
)

r_sweep = jnp.array([0.02])
outlier_prob=0.1
outlier_volume=1.0
model_box_dims = jnp.array([j.utils.aabb(m.vertices)[0] for m in renderer.meshes])

pose_proposals, weights, perfect_score = j.c2f.c2f_score_contact_parameters(
    renderer,
    0,
    obs_point_cloud_image,
    obs_point_cloud_image,
    contact_param_sweep,
    face_param_sweep,
    r_sweep,
    contact_plane_pose_in_cam_frame,
    outlier_prob,
    outlier_volume,
    model_box_dims,
)

good_poses = pose_proposals[weights[0,:] >= (perfect_score - 0.0001)]

good_poses_image = renderer.render_multiobject(good_poses, [0 for _ in range(good_poses.shape[0])])
posterior = j.viz.get_depth_image(good_poses_image[:,:,2], max=far)

depth_viz = j.get_depth_image(depth_scaled,max=intrinsics.far)
j.viz.multi_panel(
    [
        depth_viz,
        posterior,
        j.viz.overlay_image(depth_viz, posterior, alpha=0.6)
    ],
    labels=["Observed Depth", "Posterior", "Overlay"],
    label_fontsize=15
).save("overlay.png")

from IPython import embed; embed()