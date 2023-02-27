import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax
import numpy as np
import cv2
import trimesh
import time
import jax3dp3.icp
import warnings

C2F_SCHED = j.c2f.make_schedules(
    grid_widths=[0.05, 0.03, 0.02, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(7,7,21),(7,7,21),(15, 15, 1), (7,7,21)],
)


R_SWEEP = jnp.array([0.02])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0


def run_occlusion_search(image, renderer, obj_idx, timestep=1):
    intrinsics = renderer.intrinsics
    depth_scaled =  j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
    obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)

    table_plane, table_dims = j.utils.infer_table_plane(obs_point_cloud_image, image.camera_pose, intrinsics)
    occlusion_sweep = j.scene_graph.enumerate_contact_and_face_parameters(
        -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
        *(20,20,4),
        jnp.arange(6)
    )
    model_box_dims = jnp.array([j.utils.aabb(m.vertices)[0] for m in renderer.meshes])
    pose_proposals, weights, fully_occluded_weight = j.c2f.score_contact_parameters(
        renderer,
        obj_idx,
        obs_point_cloud_image,
        obs_point_cloud_image,
        occlusion_sweep,
        t3d.inverse_pose(image.camera_pose) @  table_plane,
        r_sweep,
        outlier_prob,
        outlier_volume,
        model_box_dims
    )

    good_poses = pose_proposals[weights[0,:] >= (fully_occluded_weight - 0.0001)]
    all_images_overlayed = renderer.render_multiobject(good_poses, [obj_idx for _ in range(good_poses.shape[0])])
    enumeration_viz = j.viz.resize_image(jax3dp3.viz.get_depth_image(all_images_overlayed[:,:,2], max=intrinsics.far), image.intrinsics.height, image.intrinsics.width)
    overlay_viz = jax3dp3.viz.overlay_image(rgb_viz, enumeration_viz)
    overlay_viz.save(f"occlusion_{timestep}.png")

def run_classification(image, renderer, timestep=1):
    intrinsics = renderer.intrinsics
    depth_scaled =  j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
    obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)

    table_plane, table_dims = j.utils.infer_table_plane(obs_point_cloud_image, image.camera_pose, intrinsics)
    # j.setup_visualizer()
    # j.show_cloud("1",  
    #     t3d.apply_transform(
    #         obs_point_cloud_image.reshape(-1,3),
    #         t3d.inverse_pose(table_plane) @ image.camera_pose
    #     ) 
    # )

    import jax3dp3.segment_scene
    segmentation_image, mask, viz = jax3dp3.segment_scene.segment_scene(
        image.rgb,
        obs_point_cloud_image,
        intrinsics
    )
    viz.save(f"dashboard_{timestep}.png")

    all_segmentation_ids = np.unique(segmentation_image)
    all_segmentation_ids = all_segmentation_ids[all_segmentation_ids != -1]

    for segmentation_id in all_segmentation_ids:
        depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, segmentation_image, segmentation_id, intrinsics)
        j.get_depth_image(depth_masked, max=intrinsics.far).save("masked.png")
        j.get_depth_image(depth_complement, max=intrinsics.far).save("complement.png")
        obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
        obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)

        model_box_dims = jnp.array([j.utils.aabb(m.vertices)[0] for m in renderer.meshes])
        hypotheses_over_time = j.c2f.c2f_contact_parameters(
            renderer,
            obs_point_cloud_image,
            obs_point_cloud_image_masked,
            obs_point_cloud_image_complement,
            C2F_SCHED,
            t3d.inverse_pose(image.camera_pose) @  table_plane,
            R_SWEEP,
            OUTLIER_PROB,
            OUTLIER_VOLUME,
            model_box_dims
        )

        scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
        normalized_scores = j.utils.normalize_log_scores(scores)
        order = np.argsort(-np.array(scores))

        orig_h, orig_w = image.rgb.shape[:2]
        rgb_viz = j.get_rgb_image(image.rgb)
        mask = j.utils.resize((segmentation_image == segmentation_id)* 1.0, orig_h,orig_w)[...,None]
        rgb_masked_viz = j.viz.get_rgb_image(
            image.rgb * mask
        )
        viz_images = [
            rgb_viz,
            j.viz.overlay_image(rgb_viz, rgb_masked_viz, alpha=0.6)
        ]
        top = j.viz.multi_panel(
            viz_images, 
            labels=["RGB Input", "Segment to Classify"],
            label_fontsize=50    
        )

        viz_images = []
        labels = []
        for i in order:
            (score, obj_idx, _, pose) = hypotheses_over_time[-1][i]
            depth = renderer.render_single_object(pose, obj_idx)
            depth_viz = j.viz.resize_image(j.viz.get_depth_image(depth[:,:,2], max=1.0), image.rgb.shape[0], image.rgb.shape[1])
            viz_images.append(
                j.viz.multi_panel(
                    [j.viz.overlay_image(rgb_viz, depth_viz)],
                    labels=[
                            "{:s} - {:0.2f}".format(
                            renderer.mesh_names[obj_idx],
                            normalized_scores[i]
                        )
                    ],
                    label_fontsize=50
                )
            )
        final_viz = j.viz.vstack_images(
            [top, *viz_images], border= 20
        )
        final_viz.save(f"classify_{timestep}_seg_id_{segmentation_id}.png")
    