import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time
from tqdm import tqdm
import jax

# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"


scene_name = "passive_physics_gravity_support_0001_24"
scene_name = "passive_physics_gravity_support_0001_21"
scene_name = "passive_physics_collision_0001_01"
scene_name = "passive_physics_spatio_temporal_continuity_0001_21"
scene_name = "passive_physics_object_permanence_0001_41"
scene_name = "passive_physics_shape_constancy_0001_06"
scene_path = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", scene_name +".json")

images = j.physics.load_mcs_scene_data(scene_path)
images = images

j.make_gif([j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)], "rgb.gif")

image = images[0]
intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.333)


WALL_Z = 14.5
FLOOR_Y = 1.45

point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)

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

meshes = [plane, plane]
poses = [wall_pose, floor_pose]


for i in jnp.unique(segmentation):
    point_cloud_segment = point_cloud_image[segmentation == i]
    dims, pose = j.utils.aabb(point_cloud_segment)
    if pose[:3,3][1] > FLOOR_Y - 0.01 or pose[:3,3][2] > WALL_Z - 0.01 :
        continue
    mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
        j.t3d.apply_transform(point_cloud_segment, j.t3d.inverse_pose(pose)),
        0.075
    )
    meshes.append(mesh)
    poses.append(pose)
initial_poses = jnp.array(poses).reshape(-1,4,4)



R_SWEEP = jnp.array([0.05])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0

def batched_scorer_parallel_func(poses, point_cloud_image, renderer, num_batches=3):
    all_weights = []
    for batch in jnp.array_split(poses, num_batches, 1):
        rendered_images = renderer.render_multiobject_parallel(batch, np.arange(batch.shape[0]))[...,:3]
        weights = j.threedp3_likelihood_with_r_parallel_jit(
            point_cloud_image, rendered_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
        )
        all_weights.append(weights[0,:])
    all_weights = jnp.hstack(all_weights)
    return all_weights


renderer = j.Renderer(intrinsics)
for m in meshes:
    renderer.add_mesh(m)


rerendered = renderer.render_multiobject(initial_poses, np.arange(initial_poses.shape[0]))
depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=intrinsics.far)
depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
j.vstack_images([depth_reconstruction, depth_real]).save("reconstruction.png")


d = 0.2
n = 11
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, 3
)

rotation_deltas_x = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-30.0, 30.0, 100))
rotation_deltas_y = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-30.0, 30.0, 100))
rotation_deltas_z = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(ang)))(jnp.linspace(-30.0, 30.0, 100))
rotation_deltas_global = jnp.vstack([rotation_deltas_x, rotation_deltas_y, rotation_deltas_z])
all_enumerations = np.vstack([translation_deltas_global, rotation_deltas_global])


def prior(new_pose, prev_pose, bbox_dims):
    pixel_x, pixel_y = j.project_cloud_to_pixels(prev_pose[:3,3].reshape(-1,3), intrinsics)[0]
    is_occluder = jnp.logical_or(
        (bbox_dims[0] < 0.2) * (bbox_dims[1] > 1.1),
        (bbox_dims[1] < 0.2) * (bbox_dims[0] > 1.1),
        (bbox_dims[1] > 1.1) * (bbox_dims[0] > 1.1),
    )
    offset_3d = jnp.array([0.0, 0.2, 0.0]) * (1.0 - is_occluder)
    weight = jax.scipy.stats.norm.logpdf(new_pose[:3,3] - (prev_pose[:3,3] + offset_3d), loc=0, scale=0.1).sum()
    weight += -1000.0 * ((new_pose[:3,3][1] + bbox_dims[1]/2.0) > 1.4)
    return weight

prior_parallel = jax.jit(jax.vmap(prior, in_axes=(0, None, None)))

pose_estimates = jnp.array(initial_poses)
pose_estimates_over_time = [pose_estimates]

for t in range(1,len(images)):
    image = images[t]
    print("Time: ", t)
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)

    for i in [*jnp.arange(pose_estimates.shape[0]), *jnp.arange(pose_estimates.shape[0])]:
        if i == 0 or i==1:
            continue

        pose_estimates_tiled = jnp.tile(
            pose_estimates[:,None,...], (1, all_enumerations.shape[0], 1, 1)
        )
        all_pose_proposals  = pose_estimates_tiled.at[i].set(
            jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], all_enumerations)
        )
        all_weights = batched_scorer_parallel_func(all_pose_proposals, point_cloud_image, renderer, num_batches=2)
        all_weights += prior_parallel(all_pose_proposals[i], pose_estimates[i], renderer.model_box_dims[i])
        
        pose_estimates = all_pose_proposals[:,all_weights.argmax(), :,:]

    rerendered = renderer.render_multiobject(pose_estimates, np.arange(pose_estimates.shape[0]))[...,:3]
    pixelwise_probs = j.gaussian_mixture_image_jit(point_cloud_image, rerendered, 0.05)

    segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
    segmentation_ids = jnp.unique(segmentation)

    # new_object_segmentation_ids = segmentation_ids[average_probabilities < THRESHOLD * (total_pixels >10)]
    # if len(new_object_segmentation_ids) > 0:
    #     j.get_rgb_image(image.rgb).save("rgb.png")

    for seg_id in segmentation_ids:

        num_pixels = jnp.sum(segmentation == seg_id)
        average_probability = jnp.mean(pixelwise_probs[segmentation == seg_id])



        rows, cols = jnp.where(segmentation == seg_id)
        distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
        distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())

        point_cloud_segment = point_cloud_image[segmentation == seg_id]
        dims, pose = j.utils.aabb(point_cloud_segment)

        print("Seg ID: ", seg_id, "Pixels: ",num_pixels, " Prob: ", average_probability, " dists: ", distance_to_edge_1, " ", distance_to_edge_2, " Pose: ", pose[:3, 3])

        if num_pixels < 40:
            continue

        if average_probability > 300.0:
            continue
        BUFFER = 2

        if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER:
            continue


        if pose[:3,3][1] > FLOOR_Y - 0.01 or pose[:3,3][2] > WALL_Z - 0.01 :
            continue

        j.get_depth_image(segmentation == seg_id).save(f"{t}_{seg_id}.png")
        j.get_rgb_image(image.rgb).save(f"{t}.png")
        j.get_depth_image(rerendered[:,:,2], max=WALL_Z).save(f"{t}_rerendered.png")
        j.get_depth_image(rerendered[:,:,2], max=WALL_Z).save(f"{t}_rerendered.png")

        resolution = 0.01
        voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
        min_z = voxelized[:,2].min()
        depth = voxelized[:,2].max() - voxelized[:,2].min()

        front_face = voxelized[voxelized[:,2] <= min_z+20, :]

        slices = []
        for i in range(depth):
            slices.append(front_face + jnp.array([0.0, 0.0, i]))
        full_shape = jnp.vstack(slices) * resolution


        # j.meshcat.setup_visualizer()
        # j.meshcat.clear()
        # j.meshcat.show_cloud("1", point_cloud_segment / 5.0)
        # j.meshcat.show_cloud("1", front_face * resolution / 5.0)
        # j.meshcat.show_cloud("1", full_shape / 5.0)

        dims, pose = j.utils.aabb(full_shape)

        mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
            j.t3d.apply_transform(full_shape, j.t3d.inverse_pose(pose)),
            0.075
        )
        renderer.add_mesh(mesh)
        print(f"Adding mesh! Seg ID: {seg_id} probability: {average_probability}")

        pose_estimates = jnp.vstack(
            [
                pose_estimates,
                jnp.array([pose])
            ]
        )
    
    pose_estimates_over_time.append(pose_estimates)



viz_images = []
max_objects = pose_estimates_over_time[-1].shape[0]
for t in tqdm(range(len(pose_estimates_over_time))):
    pose_estimates = pose_estimates_over_time[t]
    seg_rendered = renderer.render_multiobject(pose_estimates, np.arange(pose_estimates.shape[0]))[...,3]

    viz_images.append(j.multi_panel(
        [
            j.get_rgb_image(images[t].rgb),
            j.resize_image(
                j.get_depth_image(seg_rendered,max=max_objects+1),
                images[t].intrinsics.height,
                images[t].intrinsics.width
            )
        ],
        labels=["RGB", "Inferred"],
        title=f"{t} / {len(images)}"
    ))
j.make_gif(viz_images, f"{scene_name}.gif")

print(scene_name)

from IPython import embed; embed()

j.meshcat.setup_visualizer()


colors = np.array(j.distinct_colors(len(pose_estimates), pastel_factor=0.3))
j.meshcat.clear()
for (i,m) in enumerate(renderer.meshes):
    j.meshcat.show_trimesh(f"{i}", m, color=colors[i])

for t in tqdm(range(len(pose_estimates_over_time))):
    for i in range(len(pose_estimates_over_time[t])):
        j.meshcat.set_pose(f"{i}", pose_estimates_over_time[t][i])
    time.sleep(0.1)


o3d_viz = j.o3d_viz.O3DVis(image.intrinsics)
o3d_viz.render.scene.set_lighting(o3d_viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
viz_images = []
for t in tqdm(range(len(pose_estimates_over_time))):
    # t = 105
    o3d_viz.clear()
    pose_estimates = pose_estimates_over_time[t]
    for i in range(len(pose_estimates)):
        o3d_viz.make_trimesh(renderer.meshes[i], np.array(pose_estimates[i]), [*colors[i], 0.8])
    o3d_viz.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
    rgb = o3d_viz.capture_image(image.intrinsics, np.eye(4))
    
    viz_images.append(j.multi_panel(
        [
            j.get_rgb_image(images[t].rgb),
            j.get_rgb_image(rgb), 
        ],
        labels=["RGB", "Inferred"],
        title=f"{t} / {len(images)}"
    ))
j.make_gif(viz_images, f"{scene_name}.gif")

from IPython import embed; embed()







from IPython import embed; embed()

# Visualizations




