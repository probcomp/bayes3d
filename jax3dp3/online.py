import jax3dp3
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax
import numpy as np
import cv2
import trimesh
import time
import jax3dp3.icp

class Jax3DP3Observation(object):
    def __init__(self, rgb, depth, camera_pose, h,w,fx,fy,cx,cy,near,far):
        self.camera_params = (h,w,fx,fy,cx,cy, near, far)
        self.rgb = rgb
        self.depth = depth
        self.camera_pose = camera_pose

    def construct_from_camera_image(camera_image, near=0.001, far=5.0):
        depth = np.array(camera_image.depthPixels)
        rgb = np.array(camera_image.rgbPixels)
        camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
        K = camera_image.camera_matrix
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        near = 0.001
        return Jax3DP3Observation( rgb, depth, camera_pose, h,w,fx,fy,cx,cy,near,far)

    def construct_from_aidan_dict(d, near=0.001, far=5.0):
        depth = np.array(d["depth"] / 1000.0) 
        camera_pose = t3d.pybullet_pose_to_transform(d["extrinsics"])
        rgb = np.array(d["rgb"])
        K = d["intrinsics"][0]
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        observation = jax3dp3.Jax3DP3Observation(rgb, depth, camera_pose, h,w,fx,fy,cx,cy,near,far)
        return observation


class OnlineJax3DP3(object):
    def __init__(self):
        self.original_camera_params = None

        self.model_box_dims = jnp.zeros((0,3))
        self.mesh_names = []
        self.meshes = []

        self.table_surface_plane_pose = None
        self.table_dims = None
        self.table_pose = None

        self.contact_param_sched = None
        self.face_param_sched = None
        self.likelihood_r_sched = None

    def start_renderer(self, camera_params, scaling_factor=0.3):
        orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far = camera_params
        self.original_camera_params = (orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far)
        h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
        self.camera_params = (h,w,fx,fy,cx,cy, near, far)

        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
        self.model_box_dims = jnp.zeros((0,3))
        self.mesh_names = []
        self.meshes = []

    def add_trimesh(self, mesh, mesh_name=None, mesh_scaling_factor=1.0):
        mesh.vertices = mesh.vertices * mesh_scaling_factor
        mesh = jax3dp3.mesh.center_mesh(mesh)
        self.meshes.append(mesh)
        self.model_box_dims = jnp.vstack(
            [
                self.model_box_dims,
                jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0]
            ]
        )
        if mesh_name is None:
            num_objects = len(self.mesh_names)
            mesh_name = f"type_{num_objects}"
        self.mesh_names.append(
            mesh_name
        )
        jax3dp3.load_model(mesh)

    def process_depth_to_point_cloud_image(self, sensor_depth):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        depth = np.array(sensor_depth)
        depth = jax3dp3.utils.resize(depth, h, w)
        depth[depth > far] = far
        depth[depth < near] = far
        point_cloud_image = jnp.array(t3d.depth_to_point_cloud_image(depth, fx, fy, cx, cy))
        return point_cloud_image

    def process_segmentation_mask(self, segmentation):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        seg = np.array(segmentation)
        seg = jax3dp3.utils.resize(seg, h, w)
        return seg
    
    def segment_scene(self, rgb_original, point_cloud_image,
        viz=True
    ):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params

        import jax3dp3.segment_scene
        foreground_mask = jax3dp3.segment_scene.get_foreground_mask(rgb_original)[...,None]
        foreground_mask_scaled = jax3dp3.utils.resize(np.array(foreground_mask), h,w)
        

        point_cloud_image_above_table = (
            point_cloud_image * 
            foreground_mask_scaled[..., None]
        )

        segmentation_image = jax3dp3.utils.segment_point_cloud_image(
            point_cloud_image_above_table, threshold=0.02, min_points_in_cluster=30
        )

        viz_image = None
        if viz:
            viz_image = jax3dp3.viz.multi_panel(
                [
                    jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original, 255.0),h,w),
                    jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original * foreground_mask, 255.0),h,w),
                    jax3dp3.viz.get_depth_image(point_cloud_image[:,:,2],  max=far),
                    jax3dp3.viz.get_depth_image(point_cloud_image_above_table[:,:,2],  max=far),
                    jax3dp3.viz.get_depth_image(segmentation_image + 1, max=segmentation_image.max() + 1),
                ],
                labels=["RGB", "RGB Masked", "Depth", "Above Table", "Segmentation. {:d}".format(int(segmentation_image.max() + 1))],
            )
        return segmentation_image, viz_image

    def inference_for_segment(
        self,
        rgb_original,
        obs_point_cloud_image,
        segmentation_image,
        segmentation_id,
        object_ids_to_estimate,
        camera_pose,
        r_sweep,
        outlier_prob = 0.001,
        outlier_volume = 1.0**3,
        viz=True
    ):
        state = self
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        obs_image_masked, obs_image_complement = jax3dp3.get_image_masked_and_complement(
            obs_point_cloud_image, segmentation_image, segmentation_id, far
        )
        contact_init = self.infer_initial_contact_parameters(
            obs_image_masked, camera_pose
        )

        latent_hypotheses = []
        for obj_idx in object_ids_to_estimate:
            latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

        start = time.time()
        hypotheses_over_time = jax3dp3.c2f.c2f_contact_parameters(
            latent_hypotheses,
            self.contact_param_sched,
            self.face_param_sched,
            r_sweep,
            jnp.linalg.inv(camera_pose) @ self.table_surface_plane_pose,
            obs_point_cloud_image,
            obs_image_complement,
            outlier_prob,
            outlier_volume,
            self.model_box_dims
        )
        end= time.time()
        print ("Time elapsed:", end - start)

        if not viz:
            return hypotheses_over_time, None

        scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
        normalized_scores = jax3dp3.utils.normalize_log_scores(scores)

        (h,w,fx,fy,cx,cy, near, far) = state.camera_params
        orig_h, orig_w = rgb_original.shape[:2]
        images = []
        rgb_viz = jax3dp3.viz.get_rgb_image(rgb_original)
        rgb_masked_viz = jax3dp3.viz.get_rgb_image(rgb_original * jax3dp3.utils.resize((segmentation_image == segmentation_id)* 1.0, orig_h,orig_w )[...,None] ) 
        for i in range(len(hypotheses_over_time[-1])):
            (score, obj_idx, _, pose) = hypotheses_over_time[-1][i]
            depth = jax3dp3.render_single_object(pose, obj_idx)
            depth_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(depth[:,:,2], max=1.0),orig_h, orig_w)
            images.append(jax3dp3.viz.multi_panel([jax3dp3.viz.overlay_image(rgb_viz, depth_viz)],title="{:s} - {:0.4f}".format(state.mesh_names[object_ids_to_estimate[i]], normalized_scores[i])))
        return hypotheses_over_time, jax3dp3.viz.multi_panel([rgb_masked_viz, *images])
    

    def step(self, observation, timestep,
        r = 0.005,
        outlier_prob = 0.001,
        outlier_volume = 1.0**3,
        r_final = 0.001
    ):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params

        obs_point_cloud_image = self.process_depth_to_point_cloud_image(observation.depth)
        segmentation_image, dashboard_viz  = self.segment_scene(
            observation.rgb, obs_point_cloud_image
        )
        dashboard_viz.save(
            f"dashboard_{timestep}.png"
        )


        unique =  np.unique(segmentation_image)
        segmetation_ids = unique[unique != -1]


        # if len(self.model_box_dims) >=4:
        #     print("Occluded object search...")
        #     obj_idx = 3
        #     table_dims = self.table_dims
        #     contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        #         -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
        #         20, 20, 4,
        #         jnp.arange(6)
        #     )
        #     good_poses, pose_proposals, ranked_high_value_seg_ids = jax3dp3.c2f.c2f_occluded_object_pose_distribution(
        #         obj_idx,
        #         segmentation_image,
        #         contact_param_sweep,
        #         face_param_sweep,
        #         0.001,
        #         jnp.linalg.inv(observation.camera_pose) @ self.table_surface_plane_pose,
        #         obs_point_cloud_image,
        #         outlier_prob,
        #         outlier_volume,
        #         self.model_box_dims,
        #         self.camera_params,
        #         0.1
        #     )

        #     occlusion_viz = jax3dp3.c2f.c2f_occlusion_viz(
        #         good_poses,
        #         pose_proposals,
        #         ranked_high_value_seg_ids,
        #         observation.rgb,
        #         obj_idx,
        #         obs_point_cloud_image,
        #         segmentation_image,
        #         self.camera_params
        #     )
        #     occlusion_viz.save(
        #         f"occlusion_{timestep}.png"
        #     )
        # else:
        #     ranked_high_value_seg_ids = None
        ranked_high_value_seg_ids = None

        results = []
        for seg_id in segmetation_ids:
            print('seg_id:');print(seg_id)
            obs_image_masked, obs_image_complement = jax3dp3.get_image_masked_and_complement(
                obs_point_cloud_image, segmentation_image, seg_id, far
            )
            contact_init = self.infer_initial_contact_parameters(
                obs_image_masked, observation.camera_pose
            )

            object_ids_to_estimate = jnp.arange(len(self.mesh_names))
            latent_hypotheses = []
            for obj_idx in object_ids_to_estimate:
                latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

            start = time.time()
            hypotheses_over_time = jax3dp3.c2f.c2f_contact_parameters(
                latent_hypotheses,
                self.contact_param_sched,
                self.face_param_sched,
                r,
                jnp.linalg.inv(observation.camera_pose) @ self.table_surface_plane_pose,
                obs_point_cloud_image,
                obs_image_complement,
                outlier_prob,
                outlier_volume,
                self.model_box_dims
            )
            end= time.time()
            print ("Time elapsed:", end - start)
            
            probabilities = jax3dp3.c2f.get_probabilites(hypotheses_over_time[-1])

            scores = [i[0] for i in hypotheses_over_time[-1]]
            print('scores:');print(scores)
            final_scores = []
            for hyp in hypotheses_over_time[-1]:
                (_, obj_idx, _, pose) = hyp
                rendered_object_images = jnp.array([jax3dp3.render_single_object(pose, obj_idx)])
                rendered_images = jax3dp3.splice_in_object_parallel(rendered_object_images[...,:3], obs_image_complement)
                score = jax3dp3.threedp3_likelihood_parallel_jit(
                    obs_point_cloud_image, rendered_images, r_final, outlier_prob, outlier_volume
                )[0]
                final_scores.append(score)
            print('final_scores:');print(final_scores)
            exact_match_score = jax3dp3.threedp3_likelihood_parallel_jit(
                obs_point_cloud_image, jnp.array([obs_point_cloud_image]), r_final, outlier_prob, outlier_volume
            )[0]
            print('exact_match_score:');print(exact_match_score)

            UNKNOWN_OBJECT_THRESHOLD = -200.0
            known_object_score = (jnp.array(final_scores) - exact_match_score) / ((obs_image_masked[:,:,2] > 0).sum()) * 1000.0
            print(known_object_score)
            unknown_object = False
            if known_object_score.max() < UNKNOWN_OBJECT_THRESHOLD:
                print("I DONT KNOW THIS OBJECT!")
                unknown_object = True


            viz_panels = jax3dp3.c2f.c2f_viz(observation.rgb, hypotheses_over_time[-2:], self.mesh_names, self.camera_params, probabilities=probabilities)

            order = np.argsort(-probabilities)
            viz_panels_sorted = []
            scores_string = []
            for i in order:
                viz_panels_sorted.append(
                    viz_panels[i]
                )
            jax3dp3.viz.vstack_images(viz_panels_sorted).save(f"classify_{timestep}_seg_id_{seg_id}.png")

            results.append(
                (
                    obs_image_masked,
                    hypotheses_over_time,
                    probabilities * (1.0 - unknown_object),
                    seg_id
                )
            )
        return results, ranked_high_value_seg_ids

    def learn_new_object(self, observations, timestep):
        print("Object learning time")
        all_clouds = []
        indices = range(len(observations))
        for i in indices:
            observation = observations[i]

            obs_point_cloud_image = self.process_depth_to_point_cloud_image(observation.depth)
            segmentation_image, dashboard_viz  = self.segment_scene(
                observation.rgb, obs_point_cloud_image
            )
            dashboard_viz.save(f"shape_learning_{i}.png")

            unique = jnp.unique(segmentation_image)
            all_seg_ids = unique[unique != -1]

            segment_clouds = []
            dist_to_center = []
            for seg_id in all_seg_ids: 
                cloud = t3d.apply_transform(
                    t3d.point_cloud_image_to_points(
                        jax3dp3.get_image_masked(obs_point_cloud_image, segmentation_image, seg_id)
                    ),
                    observation.camera_pose
                )
                segment_clouds += [cloud]
                dist_to_center += [
                    jnp.linalg.norm(jnp.mean(cloud, axis=0) - jnp.array([0.5, 0.0, 0.0]))
                ]
            best_cloud = segment_clouds[np.argmin(dist_to_center)]
            # jax3dp3.show_cloud("c1", segment_clouds[1])
            all_clouds.append(best_cloud)

        fused_clouds_over_time = [all_clouds[0]]
        for i in range(1, len(all_clouds)):
            fused_cloud = fused_clouds_over_time[-1]
            best_transform = jax3dp3.icp.icp_open3d(all_clouds[i], fused_cloud)
            fused_cloud = jnp.vstack(
                [
                    fused_cloud,
                    t3d.apply_transform(all_clouds[i], best_transform)
                ]
            )
            fused_clouds_over_time.append(fused_cloud)

        fused_cloud = fused_clouds_over_time[-1]
        learned_mesh = jax3dp3.mesh.make_alpha_mesh_from_point_cloud(fused_cloud, 0.03)
        learned_mesh = jax3dp3.mesh.center_mesh(learned_mesh)
        jax3dp3.setup_visualizer()
        jax3dp3.show_trimesh("mesh", learned_mesh)
        jax3dp3.show_cloud("1", fused_cloud * 3.0)

        # import distinctipy        
        # colors = distinctipy.get_colors(len(all_clouds), pastel_factor=0.2)
        # jax3dp3.clear()
        # jax3dp3.show_cloud(f"2", fused_clouds_over_time[3]*3.0, color=np.array(colors[2]))
        # jax3dp3.show_cloud(f"1", jnp.vstack(all_clouds)*3.0, color=np.array(colors[1]))

        # jax3dp3.show_cloud(f"1",
        #     t3d.apply_transform(
        #         all_clouds[0],
        #         reg_p2p.transformation
        #     ) * 3.0,
        #     color=np.array(colors[1])
        # )


        print("Adding new mesh")
        self.add_trimesh(learned_mesh, mesh_name=f"lego_{len(self.model_box_dims)}")
        return learned_mesh




    def infer_table_plane(self, point_cloud_image, camera_pose, ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.008):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        point_cloud_flat = t3d.point_cloud_image_to_points(point_cloud_image)
        point_cloud_flat_not_far = point_cloud_flat[point_cloud_flat[:,2] < far, :]
        table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(
            t3d.apply_transform(point_cloud_flat_not_far, camera_pose), 
            ransac_threshold=ransac_threshold, inlier_threshold=inlier_threshold, segmentation_threshold=segmentation_threshold
        )

        table_pose_in_cam_frame = t3d.inverse_pose(camera_pose) @ table_pose
        if table_pose_in_cam_frame[2,2] > 0:
            table_pose = table_pose @ t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
            
        table_face_param = 2
        table_surface_plane_pose = jax3dp3.scene_graph.get_contact_plane(
            table_pose,
            table_dims,
            table_face_param
        )
        self.table_dims = table_dims
        self.table_pose = table_pose
        self.table_surface_plane_pose = table_surface_plane_pose

    def set_coarse_to_fine_schedules(self, grid_widths, angle_widths, grid_params):
        assert len(grid_widths) == len(angle_widths) == len(grid_params)
        self.contact_param_sched, self.face_param_sched = jax3dp3.c2f.make_schedules(
            grid_widths=grid_widths, angle_widths=angle_widths, grid_params=grid_params
        )

    def infer_initial_contact_parameters(self, image_masked, camera_pose):
        points_in_table_ref_frame =  t3d.apply_transform(
            t3d.point_cloud_image_to_points(image_masked), 
            t3d.inverse_pose(self.table_surface_plane_pose).dot(camera_pose)
        )
        point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
        points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
        center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
        return jnp.array([center_x, center_y, 0.0])
