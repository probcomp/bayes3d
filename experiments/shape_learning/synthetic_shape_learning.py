import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
import open3d as o3d

j.meshcat.setup_visualizer()
j.meshcat.clear()

model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
model_names = j.ycb_loader.MODEL_NAMES
obj_idx = 4
model_path = os.path.join(model_dir,"obj_" + f"{str(obj_idx+1).rjust(6, '0')}.ply")
mesh = j.mesh.load_mesh(model_path)

# Setup renderer
intrinsics = j.Intrinsics(
    200,
    200,
    100.0,100.0,
    100.0,100.0,
    0.001, 5.0
)
renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(model_path,scaling_factor=1.0/1000.0)

# Capture images from different angles
dist_away = 0.2
number_of_views = 5
angles = jnp.arange(number_of_views) * 2*jnp.pi / number_of_views
camera_poses = jnp.array([
    j.t3d.transform_from_pos_target_up(
        jnp.array([jnp.cos(angle) * dist_away,  jnp.sin(angle) * dist_away, dist_away * 0.5]),
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
    )
    for angle in angles
])

object_pose_in_camera_frame = jnp.einsum("aij,jk->aik", jnp.linalg.inv(camera_poses), jnp.eye(4))
depth_images = renderer.render_parallel(
    object_pose_in_camera_frame,
    0
)[...,2]

j.viz.multi_panel(
    [
        j.viz.get_depth_image(i,max=5.0)
        for i in depth_images
    ],
).save("multiviews.png")



grid = j.make_translation_grid_enumeration_3d(
    -0.1, -0.1, -0.1,
    0.1, 0.1, 0.1,
    100,100,100
)

from IPython import embed; embed()




t=0
threshold = 0.005
depth_images_modified = depth_images.at[depth_images <intrinsics.near].set(intrinsics.far)

counts_positive = jnp.zeros(grid.shape[0])
counts_negative = jnp.zeros(grid.shape[0])

assignment = j.occlusion.voxel_occupied_occluded_free_jit(
    camera_poses[t], depth_images_modified[t], grid, intrinsics,threshold
)
counts_positive = counts_positive.at[assignment > 0.5].set(counts_positive[assignment > 0.5] + 1.0)
counts_negative = counts_negative.at[assignment < 0.5].set(counts_negative[assignment < 0.5] + 1.0)

pseudocounts = 0.01
shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)

j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[shape_distrib > 0.5])
j.meshcat.show_cloud("occl", grid[shape_distrib == 0.5], color=j.RED)

translation_deltas = j.make_translation_grid_enumeration(
    -0.05, -0.05, -0.05, 0.05, 0.05, 0.05, 5,5,5
)

for t in range(1, len(depth_images_modified)):
    actual_camera_pose = camera_poses[t]

    proposals = jnp.einsum("ij,ajk->aik", actual_camera_pose, translation_deltas)
    assignments = j.occlusion.voxel_occupied_occluded_free_parallel_camera(
        proposals, depth_images_modified[t], grid, intrinsics, threshold
    )

    shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)
    probabilities = (assignments == 1.0) * shape_distrib + (assignments == 0.0) * (1.0 - shape_distrib) + (assignments == 0.5)
    total_probabilities = jnp.log(probabilities).sum(1)
    best_idx = total_probabilities.argmax()

    counts_positive = counts_positive.at[assignments[best_idx] > 0.5].set(counts_positive[assignments[best_idx] > 0.5] + 1.0)
    counts_negative = counts_negative.at[assignments[best_idx] < 0.5].set(counts_negative[assignments[best_idx] < 0.5] + 1.0)


shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)

j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[shape_distrib > 0.8] * 10.0)
j.meshcat.show_cloud("occl", grid[shape_distrib == 0.5] * 10.0, color=j.RED)
