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

# Setup renderer
h, w, fx,fy, cx,cy = (
    200,
    200,
    100.0,100.0,
    100.0,100.0
)
near,far = 0.1, 5000.0
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

# Load object models
model_box_dims = []
for path in model_paths:
    mesh = trimesh.load(path)  # 000001 to 000021
    mesh.vertices = mesh.vertices
    model_box_dims.append(jax3dp3.utils.aabb(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
model_box_dims = jnp.array(model_box_dims)

# Capture images from different angles
dist_away = 300.0
number_of_views = 8
angles = jnp.arange(number_of_views) * 2*jnp.pi / number_of_views
object_index = 1

camera_poses = []
for angle in angles:
    R = (
        t3d.rotation_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), angle) @
        t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/4)
    )
    z = jnp.cos(-jnp.pi + angle) * dist_away
    x = jnp.sin(-jnp.pi + angle) * dist_away
    camera_poses.append(
        t3d.transform_from_rot_and_pos(R, jnp.array([x, -dist_away, z]))
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
clouds_in_world_frame = []
for i in range(len(images)):
    cloud = t3d.point_cloud_image_to_points(images[i])
    clouds.append(cloud)
    clouds_in_world_frame.append(t3d.apply_transform(cloud, camera_poses[i]))
full_cloud = jnp.vstack(clouds_in_world_frame)
jax3dp3.show_cloud("full_cloud", full_cloud / 100.0)


grid = jax3dp3.make_translation_grid_enumeration_3d(
    -100.0, -100.0, -100.0,
    100.0, 100.0, 100.0,
    100,100,100
)

images_modified = images.at[images[:,:,:,2]<near,2].set(far)

idx = 2



occupied_counts = jnp.zeros_like(grid.shape[0])
free_counts = jnp.zeros_like(grid.shape[0])
for idx in range(images.shape[0]):
    occupied, occluded, free = get_occluded_occupied_free_masks(grid, camera_poses[idx],images_modified[idx,:,:,2], fx,fy,cx,cy)
    occupied_counts += occupied
    free_counts += free

jax3dp3.clear()
jax3dp3.show_cloud("occupied", grid[occupied_counts > 0, :] / 100.0)
occluded_mask = (free_counts == 0) * (occupied_counts == 0)
jax3dp3.show_cloud("occluded", grid[occluded_mask, :] / 100.0, color=np.array([1.0, 0.0, 0.0]))



occluded_cloud =  t3d.apply_transform(grid[occluded_mask], t3d.inverse_pose(camera_poses[0]))
occluded_rendered = jax3dp3.render_point_cloud(
   occluded_cloud,
    h, w, fx,fy,cx,cy, near, far, 0)
jax3dp3.viz.get_depth_image(occluded_rendered[:,:,2], max=far).save("occluded.png")

occupied_cloud =  t3d.apply_transform(grid[occupied_counts > 0], t3d.inverse_pose(camera_poses[0]))
occupied_rendered = jax3dp3.render_point_cloud(
   occupied_cloud, h, w, fx,fy,cx,cy, near, far, 1
)
jax3dp3.viz.get_depth_image(occupied_rendered[:,:,2], max=far).save("occupied.png")

jax3dp3.clear()
jax3dp3.show_cloud("1", occluded_cloud * 4.0, color=np.array([1.0, 0.0, 0.0]))
jax3dp3.show_cloud("2", occupied_cloud* 4.0, color=np.array([0.0, 0.0, 0.0]))

jax3dp3.clear()
jax3dp3.show_cloud("1", t3d.point_cloud_image_to_points(occluded_rendered) * 4.0, color=np.array([1.0, 0.0, 0.0]))
jax3dp3.show_cloud("2", t3d.point_cloud_image_to_points(occupied_rendered)* 4.0, color=np.array([0.0, 0.0, 0.0]))

from IPython import embed; embed()

occluded_visible = (occluded_rendered[:,:,2] < occupied_rendered[:,:,2]) * (occluded_rendered[:,:,2] > 0.0)
jax3dp3.viz.get_depth_image(occluded_visible, max=far).save("visible.png")

jax3dp3.clear()
jax3dp3.show_cloud("1", occluded_rendered[occluded_visible,:] * 4.0, color=np.array([1.0, 0.0, 0.0]))



occluded_rendered[occluded_visible,:]
occupied_rendered[occluded_visible,:]



pixels = jax3dp3.project_cloud_to_pixels(grid[occluded_mask], fx,fy,cx,cy).astype(jnp.int32)
pixels = jnp.unique(pixels,axis=0)
pixels = pixels[(0 <= pixels[:,0]) * (pixels[:,0] < w) * (0 <= pixels[:,1]) * (pixels[:,1] < h),:]





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
