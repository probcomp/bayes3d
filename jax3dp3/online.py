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


class Onlinej(object):
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

    def start_renderer(self, camera_params, scaling_factor=1.0):
        orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far = camera_params
        self.original_camera_params = (orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far)
        h,w,fx,fy,cx,cy = j.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
        self.camera_params = (h,w,fx,fy,cx,cy, near, far)

        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        # j.setup_renderer(h, w, fx, fy, cx, cy, near, far)
        j.setup_renderer(h, w, fx, fy, cx, cy, near, far)
        self.model_box_dims = jnp.zeros((0,3))
        self.mesh_names = []
        self.meshes = []

    def add_trimesh(self, mesh, mesh_name=None, mesh_scaling_factor=1.0):
        mesh.vertices = mesh.vertices * mesh_scaling_factor
        mesh = j.mesh.center_mesh(mesh)
        self.meshes.append(mesh)
        self.model_box_dims = jnp.vstack(
            [
                self.model_box_dims,
                j.utils.aabb(mesh.vertices)[0]
            ]
        )
        if mesh_name is None:
            num_objects = len(self.mesh_names)
            mesh_name = f"type_{num_objects}"
        self.mesh_names.append(
            mesh_name
        )
        j.load_model(mesh)

    def process_depth_to_point_cloud_image(self, sensor_depth):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        depth = np.array(sensor_depth)
        depth = j.utils.resize(depth, h, w)
        depth[depth > far] = far
        depth[depth < near] = far
        point_cloud_image = jnp.array(t3d.depth_to_point_cloud_image(depth, fx, fy, cx, cy))
        return point_cloud_image

    def process_segmentation_mask(self, segmentation):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        seg = np.array(segmentation)
        seg = j.utils.resize(seg, h, w)
        return seg

    
    def get_foreground_mask(self, rgb_original, point_cloud_image,
        viz=True
    ):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params

        import j.segment_scene
        foreground_mask = j.segment_scene.get_foreground_mask(rgb_original)

        return foreground_mask
        
    def classify_segment(
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
        obs_image_masked, obs_image_complement = j.get_image_masked_and_complement(
            obs_point_cloud_image, segmentation_image, segmentation_id, far
        )
        contact_init = self.infer_initial_contact_parameters(
            obs_image_masked, camera_pose
        )

        latent_hypotheses = []
        for obj_idx in object_ids_to_estimate:
            latent_hypotheses += [(-jnp.inf, obj_idx, contact_init, None)]

        start = time.time()
        hypotheses_over_time = j.c2f.c2f_contact_parameters(
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
        normalized_scores = j.utils.normalize_log_scores(scores)

        exact_match_score = j.threedp3_likelihood_parallel_jit(
            obs_point_cloud_image, jnp.array([obs_point_cloud_image]), r_sweep[0], outlier_prob, outlier_volume
        )[0]
        final_scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
        known_object_scores = (jnp.array(final_scores) - exact_match_score) / ((segmentation_image == segmentation_id).sum()) * 1000.0

        (h,w,fx,fy,cx,cy, near, far) = state.camera_params
        orig_h, orig_w = rgb_original.shape[:2]
        rgb_viz = j.viz.get_rgb_image(rgb_original)
        mask = j.utils.resize((segmentation_image == segmentation_id)* 1.0, orig_h,orig_w)[...,None]
        rgb_masked_viz = j.viz.get_rgb_image(
            rgb_original * mask
        )
        images = [
            rgb_viz,
            j.viz.overlay_image(rgb_viz, rgb_masked_viz, alpha=0.6)
        ]
        top = j.viz.multi_panel(
            images, 
            labels=["RGB Input", "Segment to Classify"],
            label_fontsize=50    
        )

        scores = [i[0] for i in hypotheses_over_time[-1]]
        order = np.argsort(-np.array(scores))

        images = []
        labels = []
        for i in order:
            (score, obj_idx, _, pose) = hypotheses_over_time[-1][i]
            depth = j.render_single_object(pose, obj_idx)
            depth_viz = j.viz.resize_image(j.viz.get_depth_image(depth[:,:,2], max=1.0),orig_h, orig_w)
            images.append(
                j.viz.multi_panel(
                    [j.viz.overlay_image(rgb_viz, depth_viz)],
                    labels=[
                         "{:s} - {:0.2f}".format(
                            state.mesh_names[object_ids_to_estimate[i]],
                            normalized_scores[i]
                        )
                    ],
                    label_fontsize=50
                )
            )
        final_viz = j.viz.vstack_images(
            [top, *images], border= 20
        )
        return hypotheses_over_time, known_object_scores, obs_image_masked, final_viz
    
    def occluded_object_search(
        self,
        rgb_original,
        obs_point_cloud_image,
        obj_idx,
        camera_pose,
        r,
        segmentation_image,
        grid_params,
        outlier_prob = 0.001,
        outlier_volume = 1.0**3,
        viz=True
    ):

        table_dims = self.table_dims
        contact_param_sweep, face_param_sweep = j.scene_graph.enumerate_contact_and_face_parameters(
            -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
            *grid_params,
            jnp.arange(6)
        )
        good_poses, pose_proposals, ranked_high_value_seg_ids = j.c2f.c2f_occluded_object_pose_distribution(
            obj_idx,
            segmentation_image,
            contact_param_sweep,
            face_param_sweep,
            r,
            t3d.inverse_pose(camera_pose) @ self.table_surface_plane_pose,
            obs_point_cloud_image,
            outlier_prob,
            outlier_volume,
            self.model_box_dims,
            self.camera_params,
        )

        if not viz:
            return good_poses, None

        occlusion_viz = j.c2f.c2f_occlusion_viz(
            good_poses,
            pose_proposals,
            ranked_high_value_seg_ids,
            rgb_original,
            obj_idx,
            obs_point_cloud_image,
            segmentation_image,
            self.camera_params
        )

        return good_poses, occlusion_viz

    def setup_on_initial_frame(self, observation, meshes, mesh_names):
        state = self
        state.start_renderer(observation.camera_params, scaling_factor=0.3)
        point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
        state.infer_table_plane(point_cloud_image, observation.camera_pose)

        state.set_coarse_to_fine_schedules(
            grid_widths=[0.15, 0.01, 0.04, 0.02],
            angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
            grid_params=[(7,7,21),(7,7,21),(15, 15, 1), (7,7,21)],
        )

        for (mesh, mesh_name) in zip(meshes, mesh_names):
            state.add_trimesh(mesh, mesh_name)

    def step(self,
        observation, timestep,
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

        object_ids_to_estimate = jnp.arange(len(self.model_box_dims))
    
        results = []
        for seg_id in segmetation_ids:
            print('seg_id:');print(seg_id)
            hypotheses_over_time, known_object_scores, obs_image_masked, inference_viz = self.classify_segment(
                observation.rgb,
                obs_point_cloud_image,
                segmentation_image,
                seg_id,
                object_ids_to_estimate,
                observation.camera_pose,
                jnp.array([0.005]),
                outlier_prob=0.2,
                outlier_volume=1.0,
                viz=True
            )

            UNKNOWN_OBJECT =  known_object_scores.max() < -120.0
            if UNKNOWN_OBJECT:
                print("UNKNOWN OBJECT!")
            inference_viz.save(f"classify_{timestep}_seg_id_{seg_id}.png")

            scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
            normalized_scores = j.utils.normalize_log_scores(scores)
            
            results.append(
                (
                    obs_image_masked,
                    hypotheses_over_time,
                    normalized_scores * (1.0 - UNKNOWN_OBJECT),
                    seg_id
                )
            )
        return results, None

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
                        j.get_image_masked(obs_point_cloud_image, segmentation_image, seg_id)
                    ),
                    observation.camera_pose
                )
                segment_clouds += [cloud]
                dist_to_center += [
                    jnp.linalg.norm(jnp.mean(cloud, axis=0) - jnp.array([0.5, 0.0, 0.0]))
                ]
            best_cloud = segment_clouds[np.argmin(dist_to_center)]
            # j.show_cloud("c1", segment_clouds[1])
            all_clouds.append(best_cloud)

        fused_clouds_over_time = [all_clouds[0]]
        for i in range(1, len(all_clouds)):
            fused_cloud = fused_clouds_over_time[-1]
            best_transform = j.icp.icp_open3d(all_clouds[i], fused_cloud)
            fused_cloud = jnp.vstack(
                [
                    fused_cloud,
                    t3d.apply_transform(all_clouds[i], best_transform)
                ]
            )
            fused_clouds_over_time.append(fused_cloud)


        j.setup_visualizer()
        
        fused_cloud = fused_clouds_over_time[-1]
        resolution = 0.005
        fused_cloud = np.rint(fused_cloud / 0.005) * 0.005
        j.show_cloud("1", fused_cloud * 3.0)
        uniq, counts = np.unique(fused_cloud, axis=0, return_counts=True)
        uniq = uniq[counts > 1]

        labels  = j.utils.segment_point_cloud(uniq, threshold=0.01)
        uniq = uniq[labels == j.utils.get_largest_cluster_id_from_segmentation(labels)]
        j.clear()
        j.show_cloud("1", uniq * 3.0)

        fused_cloud = uniq

        import open3d as o3d
        box = o3d.geometry.TriangleMesh.create_box(resolution,resolution,resolution)
        cube_vertices = np.array(box.vertices) - np.array([resolution,resolution,resolution])
        cube_faces = np.array(box.triangles)
        
        all_vertices = []
        all_faces = []
        for (i,r) in enumerate(fused_cloud):
            all_vertices.append(cube_vertices + r)
            all_faces.append(cube_faces + 8*i)
        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)
        learned_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)

        j.show_trimesh("mesh", learned_mesh)

        # learned_mesh = j.mesh.make_alpha_mesh_from_point_cloud(fused_cloud, 0.01)
        # learned_mesh = j.mesh.center_mesh(learned_mesh)
        # j.show_trimesh("mesh", learned_mesh)


        # import distinctipy        
        # colors = distinctipy.get_colors(len(all_clouds), pastel_factor=0.2)
        # j.clear()
        # j.show_cloud(f"2", fused_clouds_over_time[3]*3.0, color=np.array(colors[2]))
        # j.show_cloud(f"1", jnp.vstack(all_clouds)*3.0, color=np.array(colors[1]))

        # j.show_cloud(f"1",
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
        table_pose, table_dims = j.utils.find_table_pose_and_dims(
            t3d.apply_transform(point_cloud_flat_not_far, camera_pose), 
            ransac_threshold=ransac_threshold, inlier_threshold=inlier_threshold, segmentation_threshold=segmentation_threshold
        )

        table_pose_in_cam_frame = t3d.inverse_pose(camera_pose) @ table_pose
        if table_pose_in_cam_frame[2,2] > 0:
            table_pose = table_pose @ t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)
            
        table_face_param = 2
        table_surface_plane_pose = j.scene_graph.get_contact_plane(
            table_pose,
            table_dims,
            table_face_param
        )
        self.table_dims = table_dims
        self.table_pose = table_pose
        self.table_surface_plane_pose = table_surface_plane_pose

    def set_coarse_to_fine_schedules(self, grid_widths, angle_widths, grid_params):
        assert len(grid_widths) == len(angle_widths) == len(grid_params)
        self.contact_param_sched, self.face_param_sched = j.c2f.make_schedules(
            grid_widths=grid_widths, angle_widths=angle_widths, grid_params=grid_params
        )

    def infer_initial_contact_parameters(self, image_masked, camera_pose):
        points_in_table_ref_frame =  t3d.apply_transform(
            t3d.point_cloud_image_to_points(image_masked), 
            t3d.inverse_pose(self.table_surface_plane_pose).dot(camera_pose)
        )
        point_seg = j.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
        points_filtered = points_in_table_ref_frame[point_seg == j.utils.get_largest_cluster_id_from_segmentation(point_seg)]
        center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
        return jnp.array([center_x, center_y, 0.0])