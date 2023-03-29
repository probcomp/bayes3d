import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt

# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"


scene_name = "passive_physics_gravity_support_0001_24"
scene_name = "passive_physics_gravity_support_0001_21"
scene_name = "passive_physics_object_permanence_0001_41"
scene_name = "passive_physics_spatio_temporal_continuity_0001_21"
scene_name = "passive_physics_collision_0001_01"
scene_path = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", scene_name +".json")

images = j.physics.load_mcs_scene_data(scene_path)
images = images

j.make_gif([j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)], "rgb.gif")

image = images[0]
intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)


def is_non_object(bbox_dims):
    is_occluder = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
        (bbox_dims[0] < 0.1),
        (bbox_dims[1] < 0.1)),
        (bbox_dims[1] > 1.1)),
        (bbox_dims[0] > 1.1)),
        (bbox_dims[2] > 2.1)
    )
    return is_occluder


WALL_Z = 14.5
FLOOR_Y = 1.45

R_SWEEP = jnp.array([0.025])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0


d = 0.4
n = 7
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, 3
)


# rotation_deltas_x = jax.vmap(
#     lambda ang: j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-30.0, 30.0, 20))
# rotation_deltas_y = jax.vmap(
#     lambda ang: j.t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-30.0, 30.0, 20))
rotation_deltas_z = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(ang)))(jnp.linspace(-0.0, 0.0, 40))
rotation_deltas_global = jnp.vstack([rotation_deltas_z])
all_enumerations = np.vstack([translation_deltas_global])


renderer = j.Renderer(intrinsics)
pose_estimates_over_time = [jnp.zeros((0,4,4)), jnp.zeros((0,4,4))]
for t in range(2,len(images)):
    image = images[t]
    pose_estimates = jnp.array(pose_estimates_over_time[-1])

    print("Time: ", t, "  -  ", pose_estimates.shape[0])
    segmentation_original = image.segmentation
    segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
    segmentation_ids = jnp.unique(segmentation)
    point_cloud_image_original = j.t3d.unproject_depth(image.depth, image.intrinsics)
    depth = j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
    point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
    # j.get_depth_image(depth, max=intrinsics.far).save("depth.png")

    object_mask = jnp.zeros((intrinsics.height, intrinsics.width))
    object_ids = []
    for id in segmentation_ids:
        point_cloud_segment = point_cloud_image[segmentation == id]
        bbox_dims, pose = j.utils.aabb(point_cloud_segment)
        is_occluder = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
            (bbox_dims[0] < 0.1),
            (bbox_dims[1] < 0.1)),
            (bbox_dims[1] > 1.1)),
            (bbox_dims[0] > 1.1)),
            (bbox_dims[2] > 2.1)
        )
        if not is_occluder:
            object_mask += (segmentation == id)
            object_ids.append(id)

    object_mask = jnp.array(object_mask) > 0
    # if len(object_ids) > 0:
    #     j.get_depth_image(object_mask).save(f"object_mask_{t}.png")

    depth_complement = depth * (1.0 - object_mask) + intrinsics.far * (object_mask)
    # j.get_depth_image(depth_complement, max=intrinsics.far).save("depth_comlement.png")
    point_cloud_image_complement = j.t3d.unproject_depth(depth_complement, intrinsics)

    matched_ids = []
    for seg_id in object_ids:
        for known_object_id in range(len(renderer.meshes)):
            
            pose_estimates_tiled = jnp.tile(
                pose_estimates[known_object_id,None,...], (1, all_enumerations.shape[0], 1, 1)
            )
            all_pose_proposals  = pose_estimates_tiled.at[0].set(
                jnp.einsum("aij,ajk->aik", pose_estimates_tiled[0,...], all_enumerations)
            )
            from IPython import embed; embed()

            all_weights = []
            num_batches = 2
            for batch in jnp.array_split(all_pose_proposals, num_batches, 1):
                rendered_images = renderer.render_muleetiobject_parallel(batch, [known_object_id])[...,:3]
                rendered_images_spliced = j.splice_image_parallel(rendered_images, point_cloud_image_complement)
                j.get_depth_image(rendered_images[2,:,:,2], max=WALL_Z).save("img.png")
                weights = j.threedp3_likelihood_with_r_parallel_jit(
                    point_cloud_image, rendered_images_spliced, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
                )
                all_weights.append(weights[0,:])
            all_weights = jnp.hstack(all_weights)


    for seg_id in object_ids:
        if seg_id in matched_ids:
            continue
        
        num_pixels = jnp.sum(segmentation == seg_id)
        rows, cols = jnp.where(segmentation == seg_id)
        distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
        distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())

        point_cloud_segment = point_cloud_image_original[segmentation_original == seg_id]
        dims, pose = j.utils.aabb(point_cloud_segment)



        BUFFER = 2
        if num_pixels < 30:
            continue
        if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER:
            continue

        resolution = 0.01
        voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
        min_z = voxelized[:,2].min()
        depth = voxelized[:,2].max() - voxelized[:,2].min()

        front_face = voxelized[voxelized[:,2] <= min_z+20, :]

        slices = []
        for i in range(depth):
            slices.append(front_face + jnp.array([0.0, 0.0, i]))
        full_shape = jnp.vstack(slices) * resolution


        dims, pose = j.utils.aabb(full_shape)
        mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
            j.t3d.apply_transform(full_shape, j.t3d.inverse_pose(pose)),
            0.075
        )
        renderer.add_mesh(mesh)
        j.get_depth_image(segmentation_original == seg_id).save(f"{t}_{seg_id}_new_object.png")
        print("Adding mesh!  Seg ID: ", seg_id, "Pixels: ",num_pixels, " dists: ", distance_to_edge_1, " ", distance_to_edge_2, " Pose: ", pose[:3, 3])

        pose_estimates = jnp.vstack(
            [
                pose_estimates,
                jnp.array([pose])
            ]
        )
    pose_estimates_over_time.append(pose_estimates)

from IPython import embed; embed()