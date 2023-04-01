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
import glob
# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"



scene_name = "passive_physics_gravity_support_0001_26"


scene_name = "passive_physics_collision_0001_05"
scene_name = "passive_physics_collision_0001_03"
scene_name = "passive_physics_collision_0001_04"


scene_name = "passive_physics_spatio_temporal_continuity_0001_02"
scene_name = "passive_physics_spatio_temporal_continuity_0001_15"
scene_name = "passive_physics_spatio_temporal_continuity_0001_14"

scene_name = "passive_physics_spatio_temporal_continuity_0001_02"

scene_name = "passive_physics_shape_constancy_0001_06"

scene_name = "passive_physics_object_permanence_0001_29"
scene_name = "passive_physics_object_permanence_0001_01"
scene_name = "passive_physics_object_permanence_0001_02"
scene_name = "passive_physics_object_permanence_0001_03"

# scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_collision_0001_05.json")
# scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_shape_constancy_0001_06.json")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_spatio_temporal_continuity*.json")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_object_permanence*.json")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_shape_constancy*.json")

files = glob.glob(scene_regex)
files = [i.split("/")[-1] for i in files]

files = sorted(files)

for scene_name in files:
    print(scene_name)
    scene_path = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation",
    scene_name
    )

    images = j.physics.load_mcs_scene_data(scene_path)
    images = images

    # j.make_gif([j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)], "rgb.gif")

    WALL_Z = 14.5
    FLOOR_Y = 1.45

    image = images[0]
    intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)
    intrinsics = j.Intrinsics(
        intrinsics.height, intrinsics.width,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        intrinsics.near,
        WALL_Z
    )

    def get_object_mask(point_cloud_image, segmentation, segmentation_ids):
        object_mask = jnp.zeros(point_cloud_image.shape[:2])
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
        return object_ids, object_mask


    def get_new_shape_model(point_cloud_image, pixelwise_probs, segmentation, seg_id):
        num_pixels = jnp.sum(segmentation == seg_id)
        rows, cols = jnp.where(segmentation == seg_id)
        distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
        distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())
        average_probability = jnp.mean(pixelwise_probs[segmentation == seg_id])

        point_cloud_segment = point_cloud_image[segmentation == seg_id]
        dims, pose = j.utils.aabb(point_cloud_segment)



        BUFFER = 3
        if average_probability > 100.0:
            return None
        if num_pixels < 14:
            return None
        if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER:
            return None

        resolution = 0.01
        voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
        min_z = voxelized[:,2].min()
        depth = voxelized[:,2].max() - voxelized[:,2].min()

        front_face = voxelized[voxelized[:,2] <= min_z+20, :]
        slices = [front_face]
        for i in range(depth):
            slices.append(front_face + jnp.array([0.0, 0.0, i]))
        full_shape = jnp.vstack(slices) * resolution

        print("Seg ID: ", seg_id, "Prob: ", average_probability, " Pixels: ",num_pixels, " dists: ", distance_to_edge_1, " ", distance_to_edge_2, " Pose: ", pose[:3, 3])

        dims, pose = j.utils.aabb(full_shape)
        mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
            j.t3d.apply_transform(full_shape, j.t3d.inverse_pose(pose)),
            0.075
        )
        # j.meshcat.setup_visualizer()
        # j.meshcat.show_cloud("1", point_cloud_segment)
        # j.meshcat.show_cloud("1", full_shape)


        return mesh, pose



    dx  = 0.6
    dy = 0.6
    dz = 0.6
    gridding = j.make_translation_grid_enumeration(
        -dx, -dy, -dz, dx, dy, dz, 21,19,15
    )

    R_SWEEP = jnp.array([0.03])
    OUTLIER_PROB=0.1
    OUTLIER_VOLUME=1.0

    def prior3(new_state, prev_poses, bbox_dims, known_id):
        score = 0.0
        new_position = new_state[:3,3]

        velocity_vec = prev_poses[-1, known_id][:3,3] - prev_poses[-2][known_id][:3,3]
        velocity_vec = velocity_vec * jnp.array([0.9, 1.0, 0.0])
        velocity_vec +=  jnp.array([0.0, 0.025, 0.0])
        pred_new_position = prev_poses[-1][known_id][:3,3] + velocity_vec

        score = score + jax.scipy.stats.multivariate_normal.logpdf(
            new_position, pred_new_position, jnp.diag(jnp.array([0.01, 0.01, 0.01]))
        )
        score += -100.0 * ((new_position[1] + bbox_dims[known_id][1]/2.0) > 1.5)
        return score

    prior_parallel = jax.jit(jax.vmap(prior3, in_axes=(0, None, None, None)))


    OBJECT_POSES = jnp.zeros((1, 0, 4, 4))
    renderer = j.Renderer(intrinsics)
    for t in tqdm(range(1,len(images))):
        print("Time: ", t, "  -  ", OBJECT_POSES.shape[1])

        image = images[t]
        depth = j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
        point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
        
        segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
        segmentation_ids = jnp.unique(segmentation)

        object_ids, object_mask = get_object_mask(point_cloud_image, segmentation, segmentation_ids)
        depth_complement = depth * (1.0 - object_mask) + intrinsics.far * (object_mask)
        point_cloud_image_complement = j.t3d.unproject_depth(depth_complement, intrinsics)

        OBJECT_POSES = OBJECT_POSES[:t]
        OBJECT_POSES = jnp.vstack([OBJECT_POSES,OBJECT_POSES[-1][None, ...]])

        for known_id in range(OBJECT_POSES.shape[1]):
            current_pose_estimate = OBJECT_POSES[t, known_id, :, :]
            all_pose_proposals = [
                jnp.einsum("aij,jk->aik", 
                    gridding,
                    current_pose_estimate,
                )
            ]
            for seg_id in object_ids:
                _, center_pose = j.utils.aabb(point_cloud_image[segmentation==seg_id])
                all_pose_proposals.append(
                    jnp.einsum("aij,jk->aik", 
                        gridding,
                        center_pose,
                    )
                )
            all_pose_proposals = jnp.vstack(all_pose_proposals)

            all_weights = []
            for batch in jnp.array_split(all_pose_proposals,3):
                rendered_images = renderer.render_parallel(batch, known_id)[...,:3]
                rendered_images_spliced = j.splice_image_parallel(rendered_images, point_cloud_image_complement)
                weights = j.threedp3_likelihood_with_r_parallel_jit(
                    point_cloud_image, rendered_images_spliced, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
                ).reshape(-1)

                weights += prior_parallel(
                    batch, OBJECT_POSES[:t, :, :, :],  renderer.model_box_dims, known_id
                ).reshape(-1)

                all_weights.append(weights)
            all_weights = jnp.hstack(all_weights)

            best_pose = all_pose_proposals[all_weights.argmax()]
            OBJECT_POSES = OBJECT_POSES.at[t, known_id, ...].set(best_pose)

        all_current_pose_estimates = OBJECT_POSES[t]
        rerendered = renderer.render_multiobject(all_current_pose_estimates, jnp.arange(all_current_pose_estimates.shape[0]))[...,:3]
        rerendered_spliced = j.splice_image_parallel(jnp.array([rerendered]), point_cloud_image_complement)[0]

        # rgb_viz = j.get_rgb_image(images[t].rgb)
        # rerendered_viz = j.resize_image(j.get_depth_image(rerendered[:,:,2],max=WALL_Z), image.intrinsics.height, image.intrinsics.width)
        # j.multi_panel(
        #         [
        #             rgb_viz,
        #             rerendered_viz,
        #             j.overlay_image(rgb_viz, rerendered_viz)
        #         ],
        #     labels=["RGB", "Inferred", "Overlay"],
        #     title=f"{t} / {len(images)}"
        # ).save("state.png")

        # j.meshcat.setup_visualizer()
        # j.meshcat.show_cloud("1", rerendered.reshape(-1,3),color=j.RED)
        # j.meshcat.show_cloud("2", point_cloud_image.reshape(-1,3))

        pixelwise_probs = j.gaussian_mixture_image_jit(point_cloud_image, rerendered_spliced, R_SWEEP)

        for seg_id in object_ids:
            average_probability = jnp.mean(pixelwise_probs[segmentation == seg_id])
            # print(seg_id, average_probability)
            data = get_new_shape_model(point_cloud_image, pixelwise_probs, segmentation, seg_id)
            if data is None:
                continue
            mesh, pose = data
            renderer.add_mesh(mesh)
            print("Adding new mesh")
            
            new_col = jnp.tile(
                pose[None, None, ...],
                (OBJECT_POSES.shape[0], 1,1,1)
            )
            OBJECT_POSES = jnp.concatenate([OBJECT_POSES, new_col], axis=1)



    viz_images = []
    for t in tqdm(range(len(images))):
        all_current_pose_estimates = jnp.array(OBJECT_POSES[t])
        rerendered = renderer.render_multiobject(all_current_pose_estimates, jnp.arange(all_current_pose_estimates.shape[0]))
        rerendered_viz = j.resize_image(j.get_depth_image(rerendered[:,:,3],max=OBJECT_POSES.shape[1]+1), image.intrinsics.height, image.intrinsics.width)
        rgb_viz = j.get_rgb_image(images[t].rgb)
        viz = j.multi_panel(
                [
                    rgb_viz,
                    rerendered_viz,
                    j.overlay_image(rgb_viz, rerendered_viz)
                ],
            labels=["RGB", "Inferred", "Overlay"],
            title=f"{t} / {len(images)}"
        )
        viz_images.append(viz)
    j.make_gif(viz_images, f"{scene_name}.gif")
    print(scene_name)

from IPython import embed; embed()


    # if len(known_objects) > 0:
    #     all_current_pose_estimates = jnp.array([k[-1][1] for k in known_objects]).reshape(-1,4,4)
    #     rerendered = renderer.render_multiobject(all_current_pose_estimates, jnp.arange(all_current_pose_estimates.shape[0]))
    #     rerendered_viz = j.resize_image(j.get_depth_image(rerendered[:,:,2],max=WALL_Z), image.intrinsics.height, image.intrinsics.width)
    #     rgb_viz = j.get_rgb_image(images[t].rgb)
    #     viz = j.multi_panel(
    #             [
    #                 rgb_viz,
    #                 rerendered_viz,
    #                 j.overlay_image(rgb_viz, rerendered_viz)
    #             ],
    #         labels=["RGB", "Inferred", "Overaly"],
    #         title=f"{t} / {len(images)}"
    #     )
    #     viz.save(f"{t}.png")
    #     viz.save(f"0.png") 