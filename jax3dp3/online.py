import jax3dp3
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax
import numpy as np

class Jax3DP3Perception(object):
    def __init__(self):
        self.model_box_dims = jnp.zeros((0,3))
        self.mesh_names = []
        self.meshes = []
        self.original_camera_params = None
        self.camera_params = None

        self.table_surface_plane_pose = None
        self.table_dims = None
        self.table_pose = None

        self.contact_param_sched = None
        self.face_param_sched = None
        self.likelihood_r_sched = None


    def set_camera_params(self, orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far, scaling_factor=1.0):
        self.original_camera_params = (orig_h, orig_w, orig_fx, orig_fy, orig_cx, orig_cy, near, far)
        h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
        self.camera_params = (h,w,fx,fy,cx,cy, near, far)
        jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

    def process_depth_to_point_cloud_image(self, sensor_depth):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        depth = np.array(sensor_depth)
        depth = jax3dp3.utils.resize(depth, h, w)
        depth[depth > far] = 0.0
        point_cloud_image = jnp.array(t3d.depth_to_point_cloud_image(depth, fx, fy, cx, cy))
        return point_cloud_image

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


    def infer_table_plane(self, point_cloud_image, camera_pose, ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.008):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        point_cloud_flat = t3d.point_cloud_image_to_points(point_cloud_image)
        point_cloud_flat_not_far = point_cloud_flat[point_cloud_flat[:,2] < far, :]
        table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(
            t3d.apply_transform(point_cloud_flat_not_far, camera_pose), 
            ransac_threshold=ransac_threshold, inlier_threshold=inlier_threshold, segmentation_threshold=segmentation_threshold
        )

        table_face_param = 2
        table_surface_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)
        self.table_surface_plane_pose = table_surface_plane_pose
        self.table_dims = table_dims
        self.table_pose = table_pose

    def segment_scene(self, rgb_original, point_cloud_image, camera_pose, viz=True):
        (h,w,fx,fy,cx,cy, near, far) = self.camera_params
        point_cloud_image_above_table = (
            point_cloud_image * 
            (t3d.apply_transform(
                point_cloud_image,
                t3d.inverse(self.table_surface_plane_pose).dot(camera_pose))[:,:,2] >
              0.005)[:,:,None]
        )

        segmentation_image = jax3dp3.utils.segment_point_cloud_image(
            point_cloud_image_above_table, threshold=0.02, min_points_in_cluster=30
        )
        if viz:
            jax3dp3.viz.multi_panel(
                [
                    jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original, 255.0),h,w),
                    jax3dp3.viz.get_depth_image(point_cloud_image[:,:,2],  max=far),
                    jax3dp3.viz.get_depth_image(point_cloud_image_above_table[:,:,2],  max=far),
                    jax3dp3.viz.get_depth_image(segmentation_image + 1, max=segmentation_image.max() + 1),
                ],
                labels=["RGB", "Depth", "Above Table", "Segmentation"],
                bottom_text="Intrinsics {:d} {:d} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}\n".format(h,w,fx,fy,cx,cy,near,far),
            ).save("dashboard.png")
        return point_cloud_image_above_table, segmentation_image

    def run_detection(
            self,
            point_cloud_image, 
            point_cloud_image_above_table,
            segmentation_image,
            segmentation_id,
            camera_pose,
            outlier_prob,
            outlier_volume,
            top_k = 5,
        ):

        image_masked = point_cloud_image_above_table * (segmentation_image == segmentation_id)[:,:,None]
        image_masked_complement = point_cloud_image * (segmentation_image != segmentation_id)[:,:,None]

        points_in_table_ref_frame =  t3d.apply_transform(
            t3d.point_cloud_image_to_points(image_masked), 
            t3d.inverse(self.table_surface_plane_pose).dot(camera_pose)
        )
        point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
        points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
        center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
            
        results = jax3dp3.c2f.c2f_contact_parameters(
            jnp.array([center_x, center_y, 0.0]),
            self.contact_param_sched, self.face_param_sched, likelihood_r_sched=self.likelihood_r_sched,
            contact_plane_pose=jnp.linalg.inv(camera_pose) @ self.table_surface_plane_pose,
            gt_image_masked=image_masked, gt_img_complement=image_masked_complement,
            model_box_dims=self.model_box_dims,
            outlier_prob=outlier_prob,
            outlier_volume=outlier_volume,
            top_k=top_k
        )


        return results, image_masked, image_masked_complement

    def set_coarse_to_fine_schedules(self, grid_widths, grid_params, likelihood_r_sched):
        self.contact_param_sched, self.face_param_sched = jax3dp3.c2f.make_schedules(
            grid_widths=grid_widths, grid_params=grid_params
        )
        self.likelihood_r_sched = [0.2, 0.15, 0.1, 0.04, 0.02]

    def occlusion_search(self, obj_idx, point_cloud_image, camera_pose):
        table_dims = self.table_dims
        c, f = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
            -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, jnp.pi*2, 
            20, 20, 4,
            jnp.arange(6)
        )
        pose_proposals = jax3dp3.scene_graph.pose_from_contact_and_face_params_parallel_jit(
            c,
            f,
            self.model_box_dims[obj_idx],
            jnp.linalg.inv(camera_pose) @ self.table_surface_plane_pose
        )
        images_unmasked = jax3dp3.render_parallel(pose_proposals, obj_idx)
        image_combined_with_gt = jax.vmap(
            jax3dp3.combine_rendered_with_groud_truth, in_axes=(0, None))(
                images_unmasked, point_cloud_image
        )
        weights = jax3dp3.threedp3_likelihood_parallel_jit(
            point_cloud_image, image_combined_with_gt, 0.02, 0.0001, 20**3
        )

        best_weight = weights.max()
        good_scoring_indices = weights >= (best_weight - 1.0)
        good_poses = pose_proposals[good_scoring_indices]
        return good_poses