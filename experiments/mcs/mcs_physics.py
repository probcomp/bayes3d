import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time

j.meshcat.setup_visualizer()

# controller = mcs.create_controller(
#     os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
# )

# scene_data = mcs.load_scene_json_file(os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "charlie_0002_04_B1_debug.json"))

# step_metadata = controller.start_scene(scene_data)
# image = j.RGBD.construct_from_step_metadata(step_metadata)

# step_metadatas = [step_metadata]
# for _ in tqdm(range(200)):
#     step_metadata = controller.step("Pass")
#     if step_metadata is None:
#         break
#     step_metadatas.append(step_metadata)

# all_images = []
# for i in tqdm(range(len(step_metadatas))):
#     all_images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))

# images = all_images
# images = images[94:]

images = np.load("images.npz",allow_pickle=True)["arr_0"]
image = images[0]

intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)
renderer = j.Renderer(intrinsics)

WALL_Z = 14.5
FLOOR_Y = 1.45


point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
in_scene_mask = (point_cloud_image[:,:,2] < WALL_Z-0.05) * (point_cloud_image[:,:,1] < FLOOR_Y - 0.05)
j.viz.get_depth_image(in_scene_mask, max=1.0).save("mask.png")

# segmentation = j.utils.segment_point_cloud_image(point_cloud_image * in_scene_mask[...,None], threshold=0.1)
# j.viz.get_depth_image(segmentation+1, max=segmentation.max()+1).save("seg.png")

segmentation = (j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width) + 1) * (in_scene_mask > 0) - 1
objects = []
for i in jnp.unique(segmentation):
    if i == -1:
        continue
    objects.append(point_cloud_image[segmentation == i])
print(len(objects))

initial_object_poses = []
meshes = []
for o in objects:
    dims, pose = j.utils.aabb(o)
    initial_object_poses.append(pose)
    mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
        j.t3d.apply_transform(o, j.t3d.inverse_pose(pose)),
        0.075
    )
    meshes.append(mesh)

plane = j.mesh.make_cuboid_mesh([100.0, 100.0, 0.001])
wall_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.0, WALL_Z]),
    jnp.array([0.0, 0.0, 0.5]),
    jnp.array([0.0, -1.0, 0.0]),
)
floor_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, FLOOR_Y, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)
renderer.add_mesh(plane)
for m in meshes:
    renderer.add_mesh(m)

j.meshcat.setup_visualizer()

initial_poses = jnp.vstack([jnp.array([wall_pose, floor_pose]),jnp.array(initial_object_poses)])

def render_func(pose_estimates):
    return renderer.render_multiobject(pose_estimates, [0,0, *np.arange(len(pose_estimates)-2)+1,])[...,:3]

def render_func_parallel(pose_estimates):
    return renderer.render_multiobject_parallel(pose_estimates, [0,0, *np.arange(len(pose_estimates)-2)+1,])[...,:3]

rerendered = render_func(initial_poses)
j.meshcat.show_cloud("1", rerendered.reshape(-1,3))
depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=intrinsics.far)
depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
j.vstack_images([depth_reconstruction, depth_real]).save("reconstruction.png")


d = 0.3
n = 9
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, n
)

key = jax.random.PRNGKey(3)

import jax
rotation_deltas_global = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-20.0, 20.0, 100))

R_SWEEP = jnp.array([0.1])
OUTLIER_PROB=0.01
OUTLIER_VOLUME=5.0



def prior(new_pose, prev_pose):
    weight = jax.scipy.stats.norm.pdf(new_pose[:3,3] - (prev_pose[:3,3] + jnp.array([0.0, 0.2, 0.0])), loc=0, scale=0.1).sum()
    weight *= (new_pose[:3,3][1] < 1.4)
    return weight

prior_parallel = jax.jit(jax.vmap(prior, in_axes=(0, None)))

pose_estimates = initial_poses
pose_estimates_all = [pose_estimates]
for t in range(1,len(images)):
    image = images[t]
    print(t)
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)

    for i in [*jnp.arange(pose_estimates.shape[0]), *jnp.arange(pose_estimates.shape[0])]:
        if i == 0 or i==1:
            continue
        pose_estimates_tiled = jnp.tile(
            pose_estimates[:,None,...], (1, translation_deltas_global.shape[0], 1, 1)
        )
        pose_proposals_translation = pose_estimates_tiled.at[i].set(
            jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], translation_deltas_global)
        )
        
        pose_estimates_tiled = jnp.tile(pose_estimates[:,None,...], (1, rotation_deltas_global.shape[0], 1, 1))
        pose_proposals_rotations = pose_estimates_tiled.at[i].set(jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], rotation_deltas_global))

        all_pose_proposals = [pose_proposals_translation, pose_proposals_rotations]
        all_weights = []
        for p in all_pose_proposals:
            rendered_images = render_func_parallel(p)
            weights = j.threedp3_likelihood_with_r_parallel_jit(
                point_cloud_image, rendered_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
            )
            all_weights.append(weights[0,:])
        all_pose_proposals = jnp.concatenate(all_pose_proposals, axis=1)
        
        all_weights = jnp.hstack(all_weights)
        all_weights += prior_parallel(all_pose_proposals[i], pose_estimates[i])
        

        pose_estimates = all_pose_proposals[:,all_weights.argmax(), :,:]

    rerendered = render_func(pose_estimates[:8])
    j.get_depth_image(rerendered[:,:,2],max=WALL_Z).save("rerendered.png")
    counts = j.threedp3_counts(point_cloud_image, rerendered, R_SWEEP[0]*4)
    j.get_depth_image(counts,max=counts.max()).save("counts.png")
    unexplained = (counts == 0) * (point_cloud_image[:,:,2] < WALL_Z) * (point_cloud_image[:,:,1] < FLOOR_Y)
    j.get_depth_image(unexplained).save("unexplained.png")

    seg = j.utils.segment_point_cloud_image(point_cloud_image * unexplained[...,None], threshold=0.2, min_points_in_cluster=5)
    unique_segs = jnp.unique(seg)
    unique_segs = unique_segs[unique_segs != -1]

    if len(unique_segs) > 0:
        from IPython import embed; embed()

        BUFFER = 3
        poses = jnp.zeros((0,4,4))
        added_mesh = False
        for id in unique_segs:
            rows, cols = jnp.where(seg == id)
            if (rows < 0 + BUFFER).any() or (rows > intrinsics.height - BUFFER).any():
                continue

            if (cols < 0 + BUFFER).any() or (cols > intrinsics.width - BUFFER).any():
                continue

            o = point_cloud_image[seg == id]
            dims, pose = j.utils.aabb(o)
            poses = jnp.vstack(
                [
                    poses,
                    jnp.array([pose])
                ]
            )
            new_mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(j.t3d.apply_transform(o, j.t3d.inverse_pose(pose)), 0.075)
            print("Adding mesh!")
            added_mesh = True
            meshes.append(new_mesh)
            renderer.add_mesh(new_mesh)

        pose_estimates = jnp.concatenate([pose_estimates, jnp.array(poses)])
    pose_estimates_all.append(pose_estimates)

from IPython import embed; embed()
# assert len(images) == len(pose_estimates_all)

all_viz_images = []
for i in range(len(pose_estimates_all)):
    image = images[i]
    rerendered = render_func(pose_estimates_all[i])
    depth_reconstruction = j.resize_image(j.get_depth_image(rerendered[:,:,2], max=image.intrinsics.far), image.depth.shape[0], image.depth.shape[1])
    depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
    all_viz_images.append(j.vstack_images([depth_reconstruction, depth_real]))
j.make_gif(all_viz_images, "out.gif")


from IPython import embed; embed()


j.meshcat.setup_visualizer()

j.meshcat.clear()
colors = j.distinct_colors(len(meshes))
for i in range(len(meshes)):
    j.meshcat.show_trimesh(f"{i}",meshes[i],color=colors[i])


for t in range(len(images)):
    point_cloud_image = jnp.array(j.t3d.unproject_depth(j.utils.resize(images[t].depth, intrinsics.height, intrinsics.width), intrinsics))
    j.meshcat.show_cloud("c", point_cloud_image.reshape(-1,3))
    rerendered = render_func(pose_estimates_all[t])
    j.meshcat.show_cloud("d", rerendered.reshape(-1,3), color=j.RED)

    for i in range(2,len(pose_estimates_all[t])):
        j.meshcat.set_pose(f"{i-2}",pose_estimates_all[t][i])
    time.sleep(0.05)
