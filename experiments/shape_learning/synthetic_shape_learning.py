import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3
import open3d as o3d

jax3dp3.setup_visualizer()

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
model_names = jax3dp3.ycb_loader.MODEL_NAMES
model_paths = []
for idx in range(21):
    model_paths.append(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))

h, w, fx,fy, cx,cy = (
    200,
    200,
    100.0,100.0,
    100.0,100.0
)
near,far = 0.001, 5.0
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)


model_scaling_factor= 1.0/1000.0

model_box_dims = []
for path in model_paths:
    mesh = trimesh.load(path)  # 000001 to 000021
    mesh.vertices = mesh.vertices * model_scaling_factor
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
model_box_dims = jnp.array(model_box_dims)


dist_away = 0.3
object_index = 19

number_of_views = 4
angles = jnp.arange(number_of_views) * 2*jnp.pi / number_of_views

camera_poses = []
for angle in angles:
    R = (
        t3d.rotation_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), angle) @
        t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/6)
    )
    z = jnp.cos(-jnp.pi + angle) * dist_away
    x = jnp.sin(-jnp.pi + angle) * dist_away
    camera_poses.append(
        t3d.transform_from_rot_and_pos(R, jnp.array([x, -dist_away/4, z]))
    )
camera_poses = jnp.array(camera_poses)

object_pose = jnp.eye(4)
object_pose_in_camera_frame = jnp.einsum("aij,jk->aik", jnp.linalg.inv(camera_poses), object_pose)
images = jax3dp3.render_parallel(
    object_pose_in_camera_frame,
    object_index
)

jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.get_depth_image(i[:,:,2],max=5.0)
        for i in images
    ],
).save("multiviews.png")


clouds = []
for i in range(len(images)):
    clouds.append(
        t3d.apply_transform(t3d.point_cloud_image_to_points(images[i]), camera_poses[i])
    )

jax3dp3.show_cloud("1", np.vstack(clouds))

learned_mesh = jax3dp3.mesh.make_alpha_mesh_from_point_cloud(np.vstack(clouds), 0.01)
jax3dp3.show_trimesh("mesh", learned_mesh)


from IPython import embed; embed()


jax3dp3.clear()

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
jax3dp3.load_model(learned_mesh)

images_reconstruction = jax3dp3.render_parallel(
    object_pose_in_camera_frame,
    0
)
jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.get_depth_image(i[:,:,2],max=5.0)
        for i in images_reconstruction
    ],
).save("multiviews_reconstruction.png")

from IPython import embed; embed()
