import jax3dp3 as j
import jax.numpy as jnp
import numpy as np
import os
import trimesh

intrinsics = j.Intrinsics(
    height=1000,
    width=1000,
    fx=2000.0, fy=2000.0,
    cx=500.0, cy=500.0,
    near=0.001, far=50.0
)


j.o3d_viz.setup(intrinsics)

pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2]))
box = jnp.array([0.05, 0.04, 0.03])
j.o3d_viz.make_bounding_box(box, pose, None)


cloud = np.random.rand(100,3)
moved_cloud = j.t3d.apply_transform(cloud, pose)
j.o3d_viz.make_cloud(moved_cloud, None)


model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
offset_poses = []
for name in model_names:
    mesh_path = os.path.join(model_dir,name,"textured.obj")
    mesh_paths.append(
        mesh_path
    )

idx = 19
mesh_path  = mesh_paths[idx]

j.o3d_viz.clear()
pose = j.t3d.transform_from_rot_and_pos(
    j.t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2)
    @
    j.t3d.rotation_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), 0)
    @
    j.t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0)
    ,
    jnp.array([0.0, 0.0, 0.7]))
j.o3d_viz.make_mesh_from_file(mesh_path, None, pose=pose)

cam_pose  = j.t3d.transform_from_rot_and_pos(
    j.t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/10)
    ,
    jnp.array([0.0, -0.2, 0.0])
)
j.o3d_viz.set_camera(intrinsics, cam_pose)

rgb = np.array(j.o3d_viz.capture_image()) * 255.0
rgba = j.viz.add_rgba_dimension(rgb)
rgba = rgba.at[rgb[:,:,2] == 255,-1].set(0.0)
j.get_rgb_image(rgba).save(f"{idx}.png")

j.get_rgb_image(rgb).save("test_open3d_viz.png")







from IPython import embed; embed()