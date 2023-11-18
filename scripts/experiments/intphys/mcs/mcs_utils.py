import sys
import jax
import genjax
import bayes3d as b
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
from viz import *
from PIL import Image
import bayes3d.transforms_3d as t3d
from jax.debug import print as jprint
from tqdm import tqdm


class MCS_Observation:
    def __init__(self, rgb, depth, intrinsics, segmentation):
        """RGBD Image
        
        Args:
            rgb (np.array): RGB image
            depth (np.array): Depth image
            camera_pose (np.array): Camera pose. 4x4 matrix
            intrinsics (b.camera.Intrinsics): Camera intrinsics
            segmentation (np.array): Segmentation image
        """
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.segmentation  = segmentation

load_observations_npz = lambda x : np.load('val7_physics_npzs' + "/{}.npz".format(x),allow_pickle=True)["arr_0"]

def observations_to_data(observations, scale = 0.5):
    intrinsics_data = observations[0].intrinsics
    intrinsics = b.scale_camera_parameters(b.Intrinsics(intrinsics_data["height"],intrinsics_data["width"],
                            intrinsics_data["fx"], intrinsics_data["fy"],
                            intrinsics_data["cx"],intrinsics_data["cy"],
                            intrinsics_data["near"],intrinsics_data["far"]),scale)
    
    depths = [jax.image.resize(obs.depth, (int(obs.depth.shape[0] * scale), 
                        int(obs.depth.shape[1] * scale)), 'nearest') for obs in observations]
    segs = [jax.image.resize(obs.segmentation, (int(obs.segmentation.shape[0] * scale), 
                        int(obs.segmentation.shape[1] * scale)), 'nearest') for obs in observations]
    rgbs = [jax.image.resize(obs.rgb, (int(obs.rgb.shape[0] * scale), 
                        int(obs.rgb.shape[1] * scale), 3), 'nearest') for obs in observations]
    
    gt_images = b.unproject_depth_vmap_jit(jnp.stack(depths), intrinsics)

    return gt_images, depths, segs, rgbs, intrinsics

def fake_masker(point_cloud_image, segmentation, object_mask, object_ids, id, iter):
    object_ids = object_ids.at[iter].set(-1)
    return object_ids, object_mask

def inner_add_mask(object_mask, object_ids, segmentation, id,iter):
    object_mask += (segmentation == id)
    object_ids = object_ids.at[iter].set(id)
    return object_ids, object_mask 

def inner_fake(object_mask, object_ids, segmentation, id,iter):
    object_ids = object_ids.at[iter].set(-1)
    return object_ids, object_mask

def masker_f(point_cloud_image, segmentation, object_mask, object_ids, id, iter):

    mask = segmentation == id
    # Use the mask to select elements, keeping the original shape
    masked_point_cloud_segment = jnp.where(mask[..., None], point_cloud_image, jnp.nan)

    bbox_dims, pose = custom_aabb(masked_point_cloud_segment)
    is_occluder = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
        (bbox_dims[0] < 0.1),
        (bbox_dims[1] < 0.1)),
        (bbox_dims[1] > 1.1)),
        (bbox_dims[0] > 1.1)),
        (bbox_dims[2] > 2.1)
    )
    return jax.lax.cond(is_occluder, inner_fake, inner_add_mask,*(object_mask, object_ids, segmentation, id, iter))

def custom_aabb(object_points):
    maxs = jnp.nanmax(object_points, axis = (0,1))
    mins = jnp.nanmin(object_points, axis = (0,1))
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, t3d.transform_from_pos(center)

@jax.jit
def get_object_mask(point_cloud_image, segmentation):
    segmentation_ids = jnp.unique(segmentation, size = 10, fill_value = -1)
    object_mask = jnp.zeros(point_cloud_image.shape[:2])
    object_ids = jnp.zeros(10)
    def scan_fn(carry, id):
        object_mask, object_ids, iter = carry
        object_ids, object_mask = jax.lax.cond(id == -1, fake_masker, masker_f,*(point_cloud_image, segmentation, object_mask, object_ids, id, iter))
        return (object_mask, object_ids, iter + 1), None
    
    (object_mask, object_ids, _), _ = jax.lax.scan(scan_fn, (object_mask, object_ids, 0), segmentation_ids)
                                               
    object_mask = object_mask > 0
    return object_ids, object_mask



WALL_Z = 14.
CAM_POSE = np.array([[ 1,0,0,0],
[0,0,-1,-4.5],
[ 0,1,0,1.5],
[ 0,0,0,1]])
World2Cam = np.linalg.inv(CAM_POSE)
World2Cam[1:3] *= -1
CAM_POSE_CV2 = np.linalg.inv(World2Cam)

def in_camera_view(renderer, known_id, pose):
    """ Check if pose point is in camera view """
    # pose is assumed to be in camera frame
    return jnp.any(b.RENDERER.intrinsics.far != jnp.unique(renderer.render(pose[None,...], jnp.array([known_id]))[...,2]))

@jax.jit
def splice_image(rendered_object_image, obs_image_complement):
    keep_masks = jnp.logical_or(
        (rendered_object_image[...,2] <= obs_image_complement[..., 2]) * 
        rendered_object_image[...,2] > 0.0
        ,
        (obs_image_complement[...,2] == 0)
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images

splice_image_vmap = jax.vmap(splice_image, in_axes = (0,None))



"""
POSSIBLE FAILURE CASES:
1) The object moves so fast that it has nothing to do with the boundary
2) Min dist is not set right, either too small and the obejct misses it OR too large and other objects get counted
3) NOT A BUG but we may have to improve the quality of the object formation
"""

"""
BUGS IDENTIFIED:

Object masking goes wrong in gravity scenes with that stupid pole entering the top ---> need to mae sure this is not 
in the mask
"""

def preprocess_mcs_physics_scene(observations, MIN_DIST_THRESH = 0.6, scale = 0.1):
    # extract the data at full resolution
    gt_images, depths, segs, _, intrinsics = observations_to_data(observations,scale = 1)
    # preprocess object information before running it through model by using the gt_images
    T = gt_images.shape[0]
    review_stack = []
    init_queue = []
    review_id = 0
    registered_objects = []
    # Rule, for first frame, process any object as object model
    # Afterwards, everything must come from the edges

    get_distance = lambda x,y : jnp.linalg.norm(x[:3,3]-y[:3,3])
    print("Extracting Meshes")
    for t in tqdm(range(T)):
        # print("t = ",t)
        seg = segs[t]
        gt_image = gt_images[t]
        seg_ids = jnp.unique(seg)
        obj_ids_fixed_shape, obj_mask = get_object_mask(gt_image, seg)
        # remove all the -1 indices
        obj_ids = jnp.delete(jnp.sort(jnp.unique(obj_ids_fixed_shape)),0) # This will not be jittable
        # print(obj_ids)

        for obj_id in obj_ids:

            num_pixels = jnp.sum(seg == obj_id)
            point_cloud_segment = gt_image[seg == obj_id]
            dims, pose = b.utils.aabb(point_cloud_segment)
            rows, cols = jnp.where(seg == obj_id)
            distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height + 1).min())
            distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width + 1).min())
            # print(distance_to_edge_1, distance_to_edge_2)

            if t == 0:
                init_object_model = True
                init_object_model_metadata = {
                    't' : t,
                    'pose' : pose
                }
            else:
                init_object_model = False


            if distance_to_edge_1 == 0 or distance_to_edge_2 == 0:
                # check to ensure it is not any object in the review stack or in the init queue
                new_object = True
                if len(review_stack) > 0:
                    # check review stack first
                    distances_rs = [get_distance(pose,r['updating_pose']) for r in review_stack]
                    min_dist = np.min(distances_rs)
                    if min_dist < MIN_DIST_THRESH:
                        new_object = False
                if len(init_queue) > 0:
                    # chec init queue next
                    distances_iq = [get_distance(pose,i['updating_pose']) for i in init_queue]
                    min_dist = np.min(distances_iq)
                    init_queue_idx = distances_iq.index(min_dist)
                    if min_dist < MIN_DIST_THRESH:
                        new_object = False
                if new_object:
                    # Then this is a new object at the boundary not currently accounted for
                    #this means that object is either leaving or entering a scene (at the boundary)
                    review_id += 1
                    print("Adding review")
                    review_stack.append(
                        {
                            'id' : review_id,
                            'num_pixels' : num_pixels,
                            't' : t,
                            'distance_to_edge_1' : distance_to_edge_1,
                            'distance_to_edge_2' : distance_to_edge_2,
                            'updating_pose' : pose,
                            'init_pose' : pose
                        }
                    )
            if len(review_stack) > 0:
                # find which object under review is the closest
                distances_rs = [get_distance(pose,r['updating_pose']) for r in review_stack if r['t'] == t - 1]

                if len(distances_rs) > 0:
                    min_dist = np.min(distances_rs)
                    review_stack_idx = distances_rs.index(min_dist)
                    if min_dist < MIN_DIST_THRESH:
                        # evaluate if object is moving away or not
                        if num_pixels > review_stack[review_stack_idx]['num_pixels'] or (not (distance_to_edge_1 == 0 or distance_to_edge_2 == 0)):
                            # review passed!
                            # TODO PASS REVIEW
                            init_queue.append(review_stack[review_stack_idx])
                            print("Review passed, added to init queue")
                        else:
                            print("Review failed")
                        del review_stack[review_stack_idx]                 
                    else:
                        # Then this object must not be related to the reviewed object
                        pass
                else:
                    # Then all objects in stack were made in this time step and this object can move on
                    pass

            if len(init_queue) > 0:
                distances_iq = [get_distance(pose,i['updating_pose']) for i in init_queue]
                min_dist = np.min(distances_iq)
                init_queue_idx = distances_iq.index(min_dist)
                if min_dist < MIN_DIST_THRESH:
                    if not (distance_to_edge_1 == 0 or distance_to_edge_2 == 0):
                        # object is now ready to be initialized
                        # print("Obj init")
                        init_object_model = True
                        init_object_model_metadata = {
                            't' : init_queue[init_queue_idx]['t'],
                            'pose' : init_queue[init_queue_idx]['init_pose']  # TODO A MORE ACCURATE ESTIMATION OF INIT POSE
                        }
                        del init_queue[init_queue_idx]
                    else:
                        # this must be the object but it is still at the boundary, update the pose
                        init_queue[init_queue_idx]['updating_pose'] = pose 
                else:
                    # unrelated object
                    pass


            if init_object_model:
                # This part makes the mesh
                resolution = 0.01
                voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
                min_z = voxelized[:,2].min()
                depth_val = voxelized[:,2].max() - voxelized[:,2].min()

                front_face = voxelized[voxelized[:,2] <= min_z+20, :]
                slices = [front_face]
                for i in range(depth_val):
                    slices.append(front_face + jnp.array([0.0, 0.0, i]))
                full_shape = jnp.vstack(slices) * resolution

                dims, pose = b.utils.aabb(full_shape)
                # print("before making mesh object", get_gpu_mem())

                mesh = b.utils.make_marching_cubes_mesh_from_point_cloud(
                    b.t3d.apply_transform(full_shape, b.t3d.inverse_pose(pose)),
                    0.075
                )
                # print("before adding", get_gpu_mem())
                # renderer.add_mesh(mesh)
                
                init_object_model_metadata['mesh'] = mesh
                registered_objects.append(init_object_model_metadata)
                print("Adding new mesh at t = {}",t)
        # Ensure the every review in the review stack has the same time step as the current review
        del_idxs = []
        for i in list(reversed(range(len(review_stack)))):
            if review_stack[i]['t'] < t:
                print("Review Stack not resolved, object may have left view, deleting review")
                del review_stack[i]

        if len(review_stack) > 0 and (not np.all([r['t'] == t for r in review_stack])):
            print("REVIEW STACK HAS NOT BEEN FULLY RESOLVED, LOGIC ERROR")

    # now extract the data at low resolution
    gt_images, depths, segs, _, intrinsics = observations_to_data(observations,scale = scale)
    gt_images_bg = []
    gt_images_obj = []
    print("Extracting downsampled data")
    for t in tqdm(range(T)):
        # print("t = ",t)
        seg = segs[t]
        gt_image = gt_images[t]
        seg_ids = jnp.unique(seg)
        obj_ids_fixed_shape, obj_mask = get_object_mask(gt_image, seg)
        # remove all the -1 indices
        obj_ids = jnp.delete(jnp.sort(jnp.unique(obj_ids_fixed_shape)),0) # This will not be jittable
        # print(obj_ids)
        depth_bg = depths[t] * (1.0 - obj_mask) + intrinsics.far * (obj_mask)
        depth_obj = depths[t] * (obj_mask) + intrinsics.far * (1.0 - obj_mask)
        gt_images_bg.append(b.t3d.unproject_depth(depth_bg, intrinsics))
        gt_images_obj.append(b.t3d.unproject_depth(depth_obj, intrinsics))

    gt_images_bg = jnp.stack(gt_images_bg)
    gt_images_obj = jnp.stack(gt_images_obj)

    return gt_images, gt_images_bg, gt_images_obj, intrinsics, registered_objects