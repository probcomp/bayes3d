import bayes3d as b
import os
from tqdm import tqdm
import machine_common_sense as mcs
import numpy as np
import jax
import functools
import jax.numpy as jnp
import zmq
import pickle5
import zlib

def in_camera_view(renderer, known_id, pose):
    """ Check if pose point is in camera view """
    # pose is assumed to be in camera frame
    return jnp.mean(renderer.render(pose[None,...], jnp.array([known_id]))[...,2]) < WALL_Z

def get_object_mask(point_cloud_image, segmentation, segmentation_ids):
    object_mask = jnp.zeros(point_cloud_image.shape[:2])
    object_ids = []
    for id in segmentation_ids:
        point_cloud_segment = point_cloud_image[segmentation == id]
        bbox_dims, pose = b.utils.aabb(point_cloud_segment)
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

def physics_prior(proposed_pose, physics_estimated_pose):
    proposed_pos = proposed_pose[:3,3]
    physics_estimated_pos = physics_estimated_pose[:3,3]
    return jax.scipy.stats.multivariate_normal.logpdf(
        proposed_pos, physics_estimated_pos, jnp.diag(jnp.array([0.02, 0.02, 0.02]))
    )

physics_prior_parallel_jit = jax.jit(jax.vmap(physics_prior, in_axes=(0, None)))
physics_prior_parallel = jax.vmap(physics_prior, in_axes=(0, None))
physics_prior_jit = jax.jit(physics_prior)

# @functools.partial(
#     jax.jit,
#     static_argnums = (4)
# )
def estimate_best_physics_pose(all_poses, bbox_dims, T, known_id, tstep = 1.0/20.0, camera_pose = None):

    camera_pose = CAM_POSE_CV2
    # Assuming all poses are in camera frame
    # PHYSICS IS IN 20FPS (from step metadata)

    # extract x-y-z positions
    # T-1 
    prev_pose = all_poses[T-1][known_id,...]
    prev_pos = prev_pose[:3,3]
    # if this object existed 2 timesteps back
    if T > 1 and all_poses[T-2].shape[0] > known_id:
        # T-2
        prev_prev_pose =  all_poses[T-2][known_id,...]
    else:
        return prev_pose
        # T-1 = T-2
        # prev_prev_pose =  all_poses[T-1][known_id,...]

    prev_prev_pos = prev_prev_pose[:3,3]

    # find X-Y-Z velocity change

    # I1 & I2 -> find simple difference in world frame + check if object 
    # is on the floor and force it to have no downward vector
    # conversions to world frame
    prev_prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_prev_pos, 1], axis = None)
    prev_pos_world = camera_pose[:3,:] @ jnp.concatenate([prev_pos, 1], axis = None)
    # Note: VELOCITY IS REAL VELOCITY TIMES 1/20 (time step)
    vel_pos_world = prev_pos_world - prev_prev_pos_world

    # artificially curb the velocity of obj in world Y axis
    vel_pos_world = vel_pos_world.at[1].set(0.25 * vel_pos_world[1])

    # if object is on ground (and has upward vel), it should not have an upwards velocity (no bounce assumption)
    vel_pos_world = jax.lax.cond(
        jnp.logical_and(
            jnp.less_equal(prev_pos_world[2] - 0.5*bbox_dims[1], 0.1 * bbox_dims[1]),
            jnp.less(0, vel_pos_world[2])
        ),
        lambda x: x.at[2].set(0),
        lambda x: x,
        vel_pos_world)

    # if object is NOT on ground (and has upward vel), it should reduce it by the right amount via gravity
    vel_pos_world = jax.lax.cond(
        jnp.less(0.1 * bbox_dims[1], prev_pos_world[2] - 0.5*bbox_dims[1]),
        lambda x: x.at[2].set(vel_pos_world[2] + 3 * (0.5 * (-9.81) * (tstep)**2)), # multiply by 5
        lambda x: x,
        vel_pos_world)    

    pred_pos_world = prev_pos_world + vel_pos_world

    # check if bottom of object is below floor (AFTER) 
    object_bottom = pred_pos_world[2] - 0.5*bbox_dims[1]
    pred_pos_world = jax.lax.cond(jnp.less_equal(object_bottom, 0.01 * bbox_dims[1]),
        lambda x: x.at[2].set(0.5*bbox_dims[1]),
        lambda x: x,
        pred_pos_world)

    pred_pos = jnp.linalg.inv(camera_pose)[:3,:] @ jnp.concatenate([pred_pos_world, 1], axis = None)
    
    # I1 -> Integrate X-Y-Z forward to current time step
    # NO ROTATION CHANGE

    physics_estimated_pose = jnp.copy(prev_pose) # orientation is the same
    physics_estimated_pose = physics_estimated_pose.at[:3,3].set(pred_pos)

    return physics_estimated_pose

WALL_Z = 14.5

CAM_POSE = np.array([[ 1,0,0,0],
[0,0,-1,-4.5],
[ 0,1,0,1.5],
[ 0,0,0,1]])
World2Cam = np.linalg.inv(CAM_POSE)
World2Cam[1:3] *= -1
CAM_POSE_CV2 = np.linalg.inv(World2Cam)

class PhysicsServer():
    def __init__(self):
        pass

    def reset(self, intrinsics):
        self.original_intrinsics = intrinsics
        intrinsics = b.camera.scale_camera_parameters(self.original_intrinsics, 0.5)
        intrinsics = b.Intrinsics(
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
        self.renderer = b.Renderer(self.intrinsics)
        self.t = 0
        self.first_appearance = []

        dx  = 0.7
        dy = 0.7
        dz = 0.7
        gridding1 = b.utils.make_translation_grid_enumeration(
            -dx, -dy, -dz, dx, dy, dz, 21,15,15
        )
        self.gridding = [gridding1]
        self.plausibility = [0.0]
        self.violation_locations = [[]]
        self.images = []
        self.activate_physics_prior = []
        self.data = []

    def update(self, image, data):

        # print("Z1: ", get_gpu_mem())

        
        self.t += 1
        self.images.append(image)

        t = self.t
        images = self.images
        intrinsics = self.intrinsics
        gridding = self.gridding
        renderer = self.renderer
        plausibility = self.plausibility[-1]

        ALL_OBJECT_POSES = self.ALL_OBJECT_POSES


        R_SWEEP = jnp.array([0.02])
        OUTLIER_PROB=0.05
        OUTLIER_VOLUME=1.0

        depth = b.utils.resize(image.depth, intrinsics.height, intrinsics.width)
        point_cloud_image = b.t3d.unproject_depth(depth, intrinsics)

        segmentation = b.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
        segmentation_ids = jnp.unique(segmentation)

        object_ids, object_mask = get_object_mask(point_cloud_image, segmentation, segmentation_ids)
        depth_complement = depth * (1.0 - object_mask) + intrinsics.far * (object_mask)
        point_cloud_image_complement = b.t3d.unproject_depth(depth_complement, intrinsics)

        OBJECT_POSES = jnp.array(ALL_OBJECT_POSES[t-1])
        pre_num_active_objects = OBJECT_POSES.shape[0]
        final_physics_weight = []
        all_pose_proposals = []
        all_weights = []
        all_weights_3dp3 = []
        all_weights_physics = []
        physics_estimated_poses = []

        # find closest known ids to each seg id
        known_id_to_seg_id = {}
        mutable_known_ids = list(range(pre_num_active_objects))
        for seg_id in object_ids:
            min_dist = 1e9
            closest_known_id = None
            _, center_pose = b.utils.aabb(point_cloud_image[segmentation==seg_id])
            for known_id in mutable_known_ids:
                dist = jnp.linalg.norm(center_pose[:3,3] - OBJECT_POSES[known_id, :3,3])
                #  must be within a reasonable distance (e.g. max grid size)
                # otherwise anything new in the scene can be snapped by another object
                if dist < 0.7 and dist < min_dist:
                    min_dist = dist
                    closest_known_id = known_id
            if closest_known_id != None:
                known_id_to_seg_id[closest_known_id] = seg_id
                mutable_known_ids.remove(closest_known_id)


        for known_id in range(pre_num_active_objects):            

            current_pose_estimate = OBJECT_POSES[known_id,...]

            for gridding_iter in range(len(gridding)):
                num_proposals = gridding[gridding_iter].shape[0]
                all_pose_proposals = [
                    jnp.einsum("aij,jk->aik", 
                        gridding[gridding_iter],
                        current_pose_estimate,
                    )
                ]

                if known_id in known_id_to_seg_id:
                    closest_seg_id = known_id_to_seg_id[known_id]
                else:
                    closest_seg_id = None
                # print("B: ", get_gpu_mem())

                if gridding_iter == 0 and closest_seg_id != None:
                    # in case existing object model reappears magically somewhere else
                    # however the physics model will never allow for this, so we deactivate it
                    _, center_pose = b.utils.aabb(point_cloud_image[segmentation==closest_seg_id])
                    # if jnp.linalg.norm(center_pose[:3,3] - current_pose_estimate[:3,3]) > (3*0.7**2)**(0.5) and in_camera_view(renderer,known_id,current_pose_estimate):
                    if in_camera_view(renderer,known_id,current_pose_estimate):
                        all_pose_proposals.append(
                            jnp.einsum("aij,jk->aik", 
                                gridding[gridding_iter],
                                center_pose,
                            )
                        )

                all_pose_proposals = jnp.vstack(all_pose_proposals)

                # Find the complement to this seg_id
                if closest_seg_id != None:
                    seg_mask = segmentation == closest_seg_id
                    obj_depth_complement = depth * (1.0 - seg_mask) + intrinsics.far * (seg_mask)
                    obj_point_cloud_image_complement = b.t3d.unproject_depth(obj_depth_complement, intrinsics)
                else:
                    obj_point_cloud_image_complement = jnp.copy(point_cloud_image_complement)
                    
                all_weights_ = []
                all_weights_3dp3_ = []
                all_weights_physics_ = []

                physics_estimated_pose = estimate_best_physics_pose(ALL_OBJECT_POSES, renderer.model_box_dims[known_id], t, known_id)
                physics_estimated_poses.append(physics_estimated_pose)

                # calculate number of array_splits needed
                num_batches = int(8 * all_pose_proposals.shape[0] / gridding[gridding_iter].shape[0])
                
                for i, batch in enumerate(jnp.array_split(all_pose_proposals,num_batches)):
                    #########################################################################################
                    # print("E{}: ".format(i+1), get_gpu_mem())
                    rendered_images = renderer.render_many(batch[:,None,...], jnp.array([known_id]))[...,:3]
                    # print("F{}: ".format(i+1), get_gpu_mem())
                    rendered_images_spliced = splice_image_parallel(rendered_images, obj_point_cloud_image_complement)
                    # print("G{}: ".format(i+1), get_gpu_mem())

                    weights_3dp3 = threedp3_likelihood_with_r_parallel_jit(
                        point_cloud_image, rendered_images_spliced, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
                    ).reshape(-1)

                    if self.activate_physics_prior[known_id]:
                        weights_physics = physics_prior_parallel_jit(
                            batch, physics_estimated_pose
                        ).reshape(-1)
                    else:
                        weights_physics = jnp.zeros(weights_3dp3.shape)

                    del rendered_images
                    del rendered_images_spliced

                    weights = weights_3dp3 + weights_physics
                    all_weights_.append(weights)
                    all_weights_3dp3_.append(weights_3dp3)
                    all_weights_physics_.append(weights_physics)


                all_weights_ = jnp.hstack(all_weights_)
                all_weights_3dp3_ = jnp.hstack(all_weights_3dp3_)
                all_weights_physics_ = jnp.hstack(all_weights_physics_)

                # dont apply physics prior to snapping segmented ids
                if all_weights_.shape[0] > num_proposals:
                    all_weights_ = all_weights_.at[num_proposals:].set(all_weights_3dp3_[num_proposals:])

                # Deactivate physics prior fully if the best pose is to snap
                if gridding_iter == 0 and all_weights_.argmax() >= num_proposals:
                    self.activate_physics_prior[known_id] = True

                current_pose_estimate = all_pose_proposals[all_weights_.argmax()]

            OBJECT_POSES = OBJECT_POSES.at[known_id].set(current_pose_estimate)
            if self.activate_physics_prior[known_id]:
                final_physics_weight.append(physics_prior_jit(current_pose_estimate, physics_estimated_pose))
            else:
                final_physics_weight.append(jnp.nan)

            all_weights.append(all_weights_)
            all_weights_3dp3.append(all_weights_3dp3_)
            all_weights_physics.append(all_weights_physics_)

        ##########################################################################################
        if pre_num_active_objects > 0:
            rerendered = renderer.render(OBJECT_POSES, jnp.arange(pre_num_active_objects))[...,:3]
        else:
            rerendered = jnp.copy(point_cloud_image)

        rerendered_spliced = splice_image_parallel(jnp.array([rerendered]), point_cloud_image_complement)[0]

        final_3dp3_weight = threedp3_likelihood_image_jit(
                        point_cloud_image, rerendered_spliced, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
                    )

        
        pixelwise_probs = gaussian_mixture_image_jit(point_cloud_image, rerendered_spliced, R_SWEEP)
        # print("N: ", get_gpu_mem())

        object_data = {}
        for seg_id in object_ids:
            object_data[int(seg_id)] = {}
            average_probability = jnp.mean(pixelwise_probs[segmentation == seg_id])
            object_data[int(seg_id)]['ave_prob'] = average_probability

            if pre_num_active_objects > 0 and average_probability > 50.0:
                continue

            num_pixels = jnp.sum(segmentation == seg_id)
            object_data[int(seg_id)]['num_pixels'] = num_pixels
            if num_pixels < 14:
                continue

            rows, cols = jnp.where(segmentation == seg_id)
            distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
            distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())
            object_data[int(seg_id)]['edge_dists'] = (distance_to_edge_1, distance_to_edge_2)
            point_cloud_segment = point_cloud_image[segmentation == seg_id]
            dims, pose = b.utils.aabb(point_cloud_segment)

            BUFFER = 3
            # ASSUMPTION: Objects cannot spontaneously appear out of the middle of an image
            # if an existing object model cannot already fit to it (obj permanence):
            if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER or (pre_num_active_objects > 0 and distance_to_edge_1 > 40 and distance_to_edge_2 > 40):
                continue

            resolution = 0.01
            voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
            min_z = voxelized[:,2].min()
            depth_val = voxelized[:,2].max() - voxelized[:,2].min()

            front_face = voxelized[voxelized[:,2] <= min_z+20, :]
            slices = [front_face]
            for i in range(depth_val):
                slices.append(front_face + jnp.array([0.0, 0.0, i]))
            full_shape = jnp.vstack(slices) * resolution

            print("Seg ID: ", seg_id, "Prob: ", average_probability, " Pixels: ",num_pixels, " dists: ", distance_to_edge_1, " ", distance_to_edge_2, " Pose: ", pose[:3, 3])

            dims, pose = b.utils.aabb(full_shape)
            # print("before making mesh object", get_gpu_mem())

            mesh = b.utils.make_marching_cubes_mesh_from_point_cloud(
                b.t3d.apply_transform(full_shape, b.t3d.inverse_pose(pose)),
                0.075
            )
            # print("before adding", get_gpu_mem())
            renderer.add_mesh(mesh)
            print("Adding new mesh")
            # print("after adding", get_gpu_mem())

            self.activate_physics_prior.append(True)

            OBJECT_POSES = jnp.concatenate([OBJECT_POSES, pose[None, ...]], axis=0)

        # print("O: ", get_gpu_mem())


        ALL_OBJECT_POSES.append(OBJECT_POSES)

        self.ALL_OBJECT_POSES = ALL_OBJECT_POSES

        POSES = jnp.array(ALL_OBJECT_POSES[t])
        post_num_active_objects = POSES.shape[0]
 
        # print("P: ", get_gpu_mem())
        data.append({
            't' : t,
            'gt_rgbd' : image,
            'gt_depth' : depth,
            'gt_seg' : segmentation,
            'gt_PCI' : point_cloud_image,
            'seg_ids' : segmentation_ids,
            'obj_ids' : object_ids,
            'object_mask' :object_mask,
            'depth_complement' : depth_complement,
            'PCI_complement' : point_cloud_image_complement,
            'all_pose_proposals' : all_pose_proposals,
            'all_weights_3dp3' : all_weights_3dp3,
            'all_weights_physics' : all_weights_physics,
            'rerendered' : rerendered,
            'rerendered_spliced' : rerendered_spliced,
            '3dp3_weights' : final_3dp3_weight,
            'physics_weights' : final_physics_weight,
            'poses' : POSES,
            'physics_estimated_poses' : physics_estimated_poses,
            'object_data' : object_data, # contains ave_prob
            'pre_num_active_objects' : pre_num_active_objects,
            'post_num_active_objects' : post_num_active_objects,
        })

        return data

    def process_message(self, message):
        (request_type, args) = message
        # print(f"received reqeust {request_type}")
        if request_type == "reset":
            (h,w,fx,fy,cx,cy,near,far) = args # (400,600, 514.2991467983065,514.2991467983065,300.0,200.0,0.009999999776482582,150.0)
            intrinsics = b.Intrinsics(
                h,w,fx,fy,cx,cy,near,far
            )
            self.reset(intrinsics)
            return True
        elif request_type == "update":
            rgb, depth, seg = args
            colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
            seg_final = seg_final_flat.reshape(seg.shape[:2])
            observation = j.RGBD(rgb, depth, jnp.eye(4), None, seg_final)
            self.data = self.update(image, self.data)
            return None
        elif request_type == "get_info":
            plausibility = get_rating_from_data(self.data)
            # remove all GPU mem
            self.renderer.cleanup()
            backend = jax.lib.xla_bridge.get_backend()
            for buf in backend.live_buffers(): 
                buf.delete()
            return plausibility
        else:
            print("I HAVE NO IDEA WHAT REQUEST YOU'RE MAKING!")


def get_rating_from_data(data):
    num_objects = data[-1]['poses'].shape[0]
    # physics data
    phy_data = []
    for i in range(num_objects):
        phy_data.append([x['physics_weights'][i] if len(x['physics_weights']) > i else np.nan for x in data])
    LR_values = [[] for _ in range(num_objects)]
    for idx in tqdm(range(len(data))):
        # find num of known objects
        num_known_objs = data[idx]['poses'].shape[0]
        # find seg_ids of objects
        seg = data[idx]['gt_seg']
        object_seg_ids = np.unique(seg[data[idx]['object_mask']])
        # find pixel centroid based pose of each segment based on inferred depth
        seg_id_to_pos = {}
        for seg_id in object_seg_ids:
            ii, jj = np.where(seg == seg_id)
            i_med = int(np.median(ii))
            j_med = int(np.median(jj))
            seg_id_to_pos[seg_id] = data[idx]['rerendered_spliced'][i_med, j_med]
        # find associations with seg_id for each known_id
        known_id_to_seg_id = {}
        mutable_known_ids = [x for x in range(num_known_objs)]
        # for each segmented object (which can be seen)
        for seg_id in object_seg_ids:
            # if we have at least one unassociated object model
            if len(mutable_known_ids) > 0:
                # inferred_pos = data[idx]['poses'][known_id,...][:3,3]
                seg_pos = seg_id_to_pos[seg_id]
                inferred_positions = [data[idx]['poses'][known_id,...][:3,3] for known_id in mutable_known_ids]
                dists = [np.linalg.norm(inferred_pos - seg_pos) for inferred_pos in inferred_positions]
                associated_known_id = mutable_known_ids[np.argmin(dists)]
                known_id_to_seg_id[associated_known_id] = seg_id
                mutable_known_ids.remove(associated_known_id)
        for known_id in range(num_objects):
            if known_id <= num_known_objs and known_id in known_id_to_seg_id:
                ob_mask = seg == known_id_to_seg_id[known_id]
                obj_density = np.mean(data[idx]['3dp3_weights'][ob_mask])
                if idx == 140:
                    print(data[idx]['3dp3_weights'][ob_mask])
                LR_values[known_id].append(obj_density)
            else:
                LR_values[known_id].append(np.nan)
    phy_mins = [np.nanmin(x) for x in phy_data]
    LR_mins = [np.nanmin(x) for x in LR_values]
    phy_fail = any([np.sum([x < 0 for x in dta]) >= 2 for dta in phy_data])
    LR_fail = any([len(x) > 7 for x in LR_values]) and any([x < -2.2 for x in LR_mins])
    output = {
        'phy_data' : phy_data,
        'LR_values' : LR_values,
        'phy_mins' : phy_mins,
        'LR_mins' : LR_mins,
        'LR_fail' : LR_fail,
        'phy_fail' : phy_fail,
        'pass' : not phy_fail and not LR_fail
    }

    if output['pass']:
        plausibility = 1.0
    else:
        plausibility = 0.0
    return plausibility

def splice_image_parallel(rendered_object_image, obs_image_complement):
    keep_masks = jnp.logical_or(
        (rendered_object_image[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * 
        rendered_object_image[:,:,:,2] > 0.0
        ,
        (obs_image_complement[:,:,2] == 0)[None, ...]
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images


    # 3dp3 

@functools.partial(
jnp.vectorize,
signature='(m)->()',
excluded=(1,2,3,4,),
)
def gausssian_mixture_per_pixel(
    ij,
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    filter_size: int,
    r
):
    dists = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    probs = (jax.scipy.stats.norm.pdf(dists, loc=0, scale=r)).prod(-1).sum()
    return probs
    
def gaussian_mixture_image(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r
):
    filter_size = 3
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    probs = gausssian_mixture_per_pixel(indices, obs_xyz, rendered_xyz_padded, filter_size, r)
    return probs

gaussian_mixture_image_jit = jax.jit(gaussian_mixture_image)

def threedp3_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
):
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    probs = gaussian_mixture_image(obs_xyz, rendered_xyz, r)
    probs_with_outlier_model = probs * (1.0 - outlier_prob) / num_latent_points   + outlier_prob / outlier_volume
    return jnp.log(probs_with_outlier_model).sum()

def threedp3_likelihood_image(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
):
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    probs = gaussian_mixture_image(obs_xyz, rendered_xyz, r)
    probs_with_outlier_model = probs * (1.0 - outlier_prob) / num_latent_points   + outlier_prob / outlier_volume
    return jnp.log(probs_with_outlier_model)

threedp3_likelihood_parallel = jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None))
threedp3_likelihood_parallel_jit = jax.jit(threedp3_likelihood_parallel)
threedp3_likelihood_jit = jax.jit(threedp3_likelihood)
threedp3_likelihood_image_jit = jax.jit(threedp3_likelihood_image)
threedp3_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(threedp3_likelihood_parallel, in_axes=(None, None, 0, None, None)),
)


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5432")
physics_server = PhysicsServer()

while True:
    #  Wait for next request from client
    print("Waiting for request...")
    message = pickle5.loads(zlib.decompress(socket.recv()))
    response = physics_server.process_message(message)
    print(f"Sent response {response}...")
    socket.send(zlib.compress(pickle5.dumps(response)))
