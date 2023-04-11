import jax3dp3 as j
import os
from tqdm import tqdm
import machine_common_sense as mcs
import numpy as np
import jax
import jax.numpy as jnp

def load_mcs_scene_data(scene_path):
    cache_dir = os.path.join(j.utils.get_assets_dir(), "mcs_cache")
    scene_name = scene_path.split("/")[-1]
    
    cache_filename = os.path.join(cache_dir, f"{scene_name}.npz")
    if os.path.exists(cache_filename):
        images = np.load(cache_filename,allow_pickle=True)["arr_0"]
    else:
        controller = mcs.create_controller(
            os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
        )

        scene_data = mcs.load_scene_json_file(scene_path)

        step_metadata = controller.start_scene(scene_data)
        image = j.RGBD.construct_from_step_metadata(step_metadata)

        step_metadatas = [step_metadata]
        while True:
            step_metadata = controller.step("Pass")
            if step_metadata is None:
                break
            step_metadatas.append(step_metadata)

        all_images = []
        for i in tqdm(range(len(step_metadatas))):
            all_images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))

        images = all_images
        np.savez(cache_filename, images)

    return images


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

def prior(new_state, prev_poses, prev_prev_poses, bbox_dims, known_id):    
    score = 0.0
    new_position = new_state[:3,3]
    bottom_of_object_y = new_position[1] + bbox_dims[known_id][1]/2.0

    prev_position = prev_poses[known_id][:3,3]
    prev_prev_position = prev_prev_poses[known_id][:3,3]

    velocity_prev = (prev_position - prev_prev_position) * jnp.array([1.0, 1.0, 0.25])
    velocity_with_gravity = velocity_prev + jnp.array([-jnp.sign(velocity_prev[0])*0.01, 0.1, 0.0])

    velocity_with_gravity2 = velocity_with_gravity * jnp.array([1.0 * (jnp.abs(velocity_with_gravity[0]) > 0.1), 1.0, 1.0 ])
    velocity = velocity_with_gravity2

    pred_new_position = prev_position + velocity

    score = score + jax.scipy.stats.multivariate_normal.logpdf(
        new_position, pred_new_position, jnp.diag(jnp.array([0.02, 0.02, 0.02]))
    )
    score += -100.0 * (bottom_of_object_y > 1.5)
    return score

prior_jit = jax.jit(prior)
prior_parallel_jit = jax.jit(jax.vmap(prior, in_axes=(0, None,  None, None, None)))

WALL_Z = 14.5
FLOOR_Y = 1.45

class PhysicsServer():
    def __init__(self):
        pass

    def reset(self, intrinsics):
        self.original_intrinsics = intrinsics
        intrinsics = j.camera.scale_camera_parameters(self.original_intrinsics, 0.25)
        intrinsics = j.Intrinsics(
            intrinsics.height, intrinsics.width,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.near,
            WALL_Z
        )
        self.intrinsics = intrinsics
        
        self.ALL_OBJECT_POSES = [jnp.zeros((0, 4, 4))]
        self.renderer = j.Renderer(self.intrinsics)
        self.t = 0
        self.first_appearance = []

        dx  = 0.7
        dy = 0.7
        dz = 0.7
        gridding1 = j.make_translation_grid_enumeration(
            -dx, -dy, -dz, dx, dy, dz, 21,15,15
        )
        self.gridding = [gridding1]
        self.plausibility = [0.0]
        self.violation_locations = [[]]
        self.images = []

    def update(self, image):
        self.t += 1
        self.images.append(image)

        t = self.t
        images = self.images
        intrinsics = self.intrinsics
        gridding = self.gridding
        renderer = self.renderer
        plausibility = self.plausibility[-1]

        ALL_OBJECT_POSES = self.ALL_OBJECT_POSES


        R_SWEEP = jnp.array([0.03])
        OUTLIER_PROB=0.05
        OUTLIER_VOLUME=1.0


        depth = j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
        point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
        
        segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
        segmentation_ids = jnp.unique(segmentation)

        object_ids, object_mask = j.physics.get_object_mask(point_cloud_image, segmentation, segmentation_ids)
        depth_complement = depth * (1.0 - object_mask) + intrinsics.far * (object_mask)
        point_cloud_image_complement = j.t3d.unproject_depth(depth_complement, intrinsics)

        OBJECT_POSES = jnp.array(ALL_OBJECT_POSES[t-1])
        for known_id in range(OBJECT_POSES.shape[0]):

            current_pose_estimate = OBJECT_POSES[known_id, :, :]

            for gridding_iter in range(len(gridding)):
                all_pose_proposals = [
                    jnp.einsum("aij,jk->aik", 
                        gridding[gridding_iter],
                        current_pose_estimate,
                    )
                ]
                if gridding_iter == 0:
                    for seg_id in object_ids:
                        _, center_pose = j.utils.aabb(point_cloud_image[segmentation==seg_id])
                        all_pose_proposals.append(
                            jnp.einsum("aij,jk->aik", 
                                gridding[gridding_iter],
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

                    if ALL_OBJECT_POSES[t-1].shape[0] != ALL_OBJECT_POSES[t-2].shape[0]:
                        prev_prev_poses =  ALL_OBJECT_POSES[t-1]
                    else:
                        prev_prev_poses =  ALL_OBJECT_POSES[t-2]


                    weights += j.physics.prior_parallel_jit(
                        batch, ALL_OBJECT_POSES[t-1],  prev_prev_poses, renderer.model_box_dims, known_id
                    ).reshape(-1)

                    all_weights.append(weights)
                all_weights = jnp.hstack(all_weights)

                current_pose_estimate = all_pose_proposals[all_weights.argmax()]

            OBJECT_POSES = OBJECT_POSES.at[known_id].set(current_pose_estimate)

        rerendered = renderer.render_multiobject(OBJECT_POSES, jnp.arange(OBJECT_POSES.shape[0]))[...,:3]
        rerendered_spliced = j.splice_image_parallel(jnp.array([rerendered]), point_cloud_image_complement)[0]
        pixelwise_probs = j.gaussian_mixture_image_jit(point_cloud_image, rerendered_spliced, R_SWEEP)

        for seg_id in object_ids:
            average_probability = jnp.mean(pixelwise_probs[segmentation == seg_id])
            print(seg_id, average_probability)

            if average_probability > 50.0:
                continue

            num_pixels = jnp.sum(segmentation == seg_id)
            if num_pixels < 14:
                continue

            rows, cols = jnp.where(segmentation == seg_id)
            distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
            distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())

            point_cloud_segment = point_cloud_image[segmentation == seg_id]
            dims, pose = j.utils.aabb(point_cloud_segment)

            BUFFER = 3

            if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER:
                continue

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
            
            renderer.add_mesh(mesh)
            print("Adding new mesh")

            OBJECT_POSES = jnp.concatenate([OBJECT_POSES, pose[None, ...]], axis=0)
        ALL_OBJECT_POSES.append(OBJECT_POSES)

        self.ALL_OBJECT_POSES = ALL_OBJECT_POSES

        violations = []

        POSES = jnp.array(ALL_OBJECT_POSES[t])

        num_objects_previous = ALL_OBJECT_POSES[t-1].shape[0]
        num_objects_now = ALL_OBJECT_POSES[t].shape[0]
        num_new_objects = num_objects_now - num_objects_previous
        if num_new_objects > 0:
            print(f"{num_new_objects} new objects!")
        for new_object_index in range(num_objects_previous, num_objects_now):
            self.first_appearance.append(t)
            position = POSES[new_object_index,:3,3]
            rerendered = renderer.render_multiobject(POSES[new_object_index:new_object_index+1], [new_object_index])
            rows, cols = jnp.where(rerendered[:,:,2] > 0.0)
            distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
            distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())     
            if distance_to_edge_1 > 15 and  distance_to_edge_2 > 15:
                pixx, pixy= np.array(j.project_cloud_to_pixels(jnp.array([ position]), intrinsics).astype(jnp.int32)[0])
                for t_ in range(t-3):
                    occluded = j.utils.resize(images[t_].depth, intrinsics.height, intrinsics.width)[pixy, pixx] < position[2]
                    if not occluded:
                        pixx, pixy= np.array(j.project_cloud_to_pixels(jnp.array([ position]), intrinsics).astype(jnp.int32)[0]) * 4
                        violations.append({"x": pixx, "y": pixy})
                        plausibility -= 0.1
                        print("Object initialize not near edge! Implausiblepix!")
                        break

        for id_1 in range(num_objects_now):
            for id_2 in range(num_objects_now):
                if id_1 != id_2:
                    distance = jnp.linalg.norm(POSES[id_1,:3,3] - POSES[id_2,:3,3])
                    if distance < 0.4:
                        pixx, pixy= np.array(j.project_cloud_to_pixels(jnp.array([ POSES[id_1,:3,3]]), intrinsics).astype(jnp.int32)[0]) * 4
                        violations.append({"x": pixx, "y": pixy})
                        plausibility -= 0.1
                        print("Objects too close together! Implausible!")


        for id in range(num_objects_now):
            z_delta = POSES[id,:3,3][1] - ALL_OBJECT_POSES[self.first_appearance[id]][id,:3,3][1]
            if z_delta < -0.2:
                pixx, pixy=  np.array(j.project_cloud_to_pixels(jnp.array([ POSES[id,:3,3]]), intrinsics).astype(jnp.int32)[0]) * 4
                violations.append({"x": pixx, "y": pixy})
                plausibility -= 0.05
                print("Objects is not obeying gravity! Implausible!")

            if POSES[id,:3,3][1] > 1.5:
                pixx, pixy= np.array(j.project_cloud_to_pixels(jnp.array([ POSES[id,:3,3]]), intrinsics).astype(jnp.int32)[0]) * 4
                violations.append({"x": pixx, "y": pixy})
                plausibility -= 0.01
                print("Objects inside floor!")

            if POSES[id,:3,3][2] > 13.0:
                pixx, pixy= np.array(j.project_cloud_to_pixels(jnp.array([ POSES[id,:3,3]]), intrinsics).astype(jnp.int32)[0]) * 4
                violations.append({"x": pixx, "y": pixy})
                plausibility -= 0.01
                print("Objects behind wall!")

        self.plausibility.append(plausibility)
        self.violation_locations.append(violations)


    def process_message(self, message):
        (request_type, args) = message
        print(f"received reqeust {request_type}")
        if request_type == "reset":
            (h,w,fx,fy,cx,cy,near,far) = args
            intrinsics = j.Intrinsics(
                h,w,fx,fy,cx,cy,near,far
            )
            self.reset(intrinsics)
            return True
        elif request_type == "update":
            rgb, depth, seg = args
            colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
            seg_final = seg_final_flat.reshape(seg.shape[:2])
            observation = j.RGBD(rgb, depth, jnp.eye(4), self.original_intrinsics, seg_final)
            self.update(observation)
            return None
        elif request_type == "get_info":
            return self.plausibility[-1], self.violation_locations[-1]
        else:
            print("I HAVE NO IDEA WHAT REQUEST YOU'RE MAKING!")
