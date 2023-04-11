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

IMG_NUMBER = 3
rgbd = rgbds[IMG_NUMBER]
gt_pose = poses[IMG_NUMBER]
original_intrinsics = rgbd.intrinsics
intrinsics = j.scale_camera_parameters(original_intrinsics, 0.1)


model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
model_names = ["obj_" + f"{str(idx+1).rjust(6, '0')}.ply" for idx in range(21)]
renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(os.path.join(model_dir,model_names[id]), scaling_factor= 1.0/1000.0)

depth = rgbd.depth
depth_scaled = j.utils.resize(depth, intrinsics.height, intrinsics.width)
point_cloud_image = j.t3d.unproject_depth(depth_scaled, intrinsics)

j.meshcat.setup_visualizer()

# configure c2f
grid_widths = [0.1, 0.05, 0.03, 0.01, 0.01, 0.01]
rot_angle_widths = [jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi/5, jnp.pi/5]
sphere_angle_widths = [jnp.pi, jnp.pi/2, jnp.pi/4, jnp.pi/4, jnp.pi/5, jnp.pi/5]
grid_params =  [(3,3,3,75*5,15), (3,3,3,75*3,21),(3,3,3,55,45),(3,3,3,55,45), (3,3,3,45,45), (3,3,3,45,45)]  # (num_x, num_y, num_z, num_fib_sphere, num_planar_angle)

scheds = j.c2f.make_schedules(
    grid_widths=grid_widths, 
    angle_widths=rot_angle_widths, 
    grid_params=grid_params, 
    full_pose=True, 
    sphere_angle_widths=sphere_angle_widths
)


j.meshcat.show_cloud("1", point_cloud_image.reshape(-1,3))

center = jnp.mean(point_cloud_image[point_cloud_image[:,:,2]< intrinsics.far],axis=0)
pose_estimate = j.t3d.transform_from_pos(center)
best_weight = -jnp.inf

for deltas in scheds:
    for batch in jnp.array_split(deltas, deltas.shape[0] // 2000):
        pose_proposals = jnp.einsum('ij,ajk->aik', pose_estimate, batch)

        rendered_depth = renderer.render_parallel(pose_proposals, 0)[...,2]
        rendered_point_cloud_images = j.t3d.unproject_depth_vmap_jit(rendered_depth, intrinsics)

        R_SWEEP = jnp.array([0.02])
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

j.meshcat.show_cloud("2", renderer.render_single_object(pose_estimate, 0)[...,:3].reshape(-1,3), color=j.RED)



import jax3dp3.posecnn_densefusion
densefusion = j.posecnn_densefusion.DenseFusion()


IMG_NUMBER = 3
rgbd = rgbds[IMG_NUMBER]
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name="1")
print(results)

# pose_proposals = jnp.array([gt_pose, gt_pose])
# rendered_depth = renderer.render_parallel(pose_proposals, 0)[...,2]
# rendered_point_cloud_images = j.t3d.unproject_depth_vmap_jit(rendered_depth, intrinsics)

# weights = j.threedp3_likelihood_with_r_parallel_jit(
#     point_cloud_image, rendered_point_cloud_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
# )
# print(weights.max())
# j.meshcat.show_cloud("2", renderer.render_single_object(jnp.array(gt_pose), 0)[...,:3].reshape(-1,3), color=j.RED)

from IPython import embed; embed()

