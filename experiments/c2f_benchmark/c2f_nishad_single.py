import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
from tqdm import tqdm
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
import jax

DATASET_FILENAME = "dataset.npz"  # npz file
DATASET_FILE = os.path.join(j.utils.get_assets_dir(), f"datasets/{DATASET_FILENAME}")

data = np.load(DATASET_FILE,allow_pickle=True)
rgbds = data["rgbds"]
poses = data["poses"]
id = data["id"].item()
IMG_NUMBER = 0
rgbd = rgbds[IMG_NUMBER]
gt_pose = poses[IMG_NUMBER]
original_intrinsics = rgbd.intrinsics
intrinsics = j.scale_camera_parameters(original_intrinsics, 0.25)


model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
model_names = ["obj_" + f"{str(idx+1).rjust(6, '0')}.ply" for idx in range(21)]
renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(os.path.join(model_dir,model_names[id]), scaling_factor= 1.0/1000.0)

depth = rgbd.depth
depth_scaled = j.utils.resize(depth, intrinsics.height, intrinsics.width)
point_cloud_image = j.t3d.unproject_depth(depth_scaled, intrinsics)

j.meshcat.setup_visualizer()
j.meshcat.show_cloud("1", point_cloud_image.reshape(-1,3))


# T_HALF_WIDTH = 0.1
# translation_grid = j.enumerations.make_translation_grid_enumeration(-T_HALF_WIDTH,-T_HALF_WIDTH,-T_HALF_WIDTH,T_HALF_WIDTH,T_HALF_WIDTH,T_HALF_WIDTH,10,10,10)

# rotation_grid = j.enumerations.get_rotation_proposals(200,20)

# angles = jnp.linspace(0.0, 2*jnp.pi, 301)
# rotations_z = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([0.0,0.0, 1.0]), angles)
# rotations_y = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([0.0,1.0, 0.0]), angles)
# rotations_x = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([1.0,0.0, 0.0]), angles)

# all_translation_grids = [translation_grid]
# all_rotation_grids = [rotations_z, rotations_y, rotations_x, rotation_grid]
# all_grids = [all_translation_grids, all_rotation_grids]


center = jnp.mean(point_cloud_image[point_cloud_image[:,:,2]< intrinsics.far],axis=0)
pose_estimate = j.t3d.transform_from_pos(center)
best_weight = -jnp.inf


for it in tqdm(range(50)):
    translation_iter = (it % 2) ==0

    for grid in all_grids[it % 2]:
        if translation_iter:
            pose_proposals = jnp.einsum('aij,jk->aik', grid, pose_estimate)
        else:
            pose_proposals = jnp.einsum('ij,ajk->aik', pose_estimate, grid)


        rendered_depth = renderer.render_parallel(pose_proposals, 0)[...,2]
        rendered_point_cloud_images = j.t3d.unproject_depth_vmap_jit(rendered_depth, intrinsics)

        R_SWEEP = jnp.array([0.05])
        OUTLIER_PROB=0.1
        OUTLIER_VOLUME=1.0
        weights = j.threedp3_likelihood_with_r_parallel_jit(
            point_cloud_image, rendered_point_cloud_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
        )
        if weights.max() > best_weight:
            pose_estimate = pose_proposals[weights.argmax()]
            best_weight = weights.max()
    print(best_weight)

j.meshcat.show_cloud("2", renderer.render_single_object(pose_estimate, 0)[...,:3].reshape(-1,3), color=j.RED)
print(weights.max())



pose_proposals = jnp.array([gt_pose, gt_pose])
rendered_depth = renderer.render_parallel(pose_proposals, 0)[...,2]
rendered_point_cloud_images = j.t3d.unproject_depth_vmap_jit(rendered_depth, intrinsics)

weights = j.threedp3_likelihood_with_r_parallel_jit(
    point_cloud_image, rendered_point_cloud_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
)
print(weights.max())
# j.meshcat.show_cloud("2", renderer.render_single_object(jnp.array(gt_pose), 0)[...,:3].reshape(-1,3), color=j.RED)

from IPython import embed; embed()

