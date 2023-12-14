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
import matplotlib.pyplot as plt
from bayes3d.viz import open3dviz as o3dviz
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from jax.scipy.special import logsumexp

def normalize_weights(log_weights):
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total_weight
    return log_normalized_weights

def ess(log_normalized_weights):
    log_ess = -logsumexp(2. * log_normalized_weights)
    return jnp.exp(log_ess)


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

def observations_to_data_by_frame(observations, frame_idx, scale = 0.5):
    intrinsics_data = observations[frame_idx].intrinsics
    if 'cam_pose' in dir(observations[frame_idx]):
        cam_pose = observations[frame_idx].cam_pose
    else:
        cam_pose = CAM_POSE_CV2

    intrinsics = b.scale_camera_parameters(b.Intrinsics(intrinsics_data["height"],intrinsics_data["width"],
                            intrinsics_data["fx"], intrinsics_data["fy"],
                            intrinsics_data["cx"],intrinsics_data["cy"],
                            intrinsics_data["near"],intrinsics_data["far"]),scale)
    
    obs = observations[frame_idx]
    depth = np.asarray(jax.image.resize(obs.depth, (int(obs.depth.shape[0] * scale), 
                        int(obs.depth.shape[1] * scale)), 'nearest'))
    seg = np.asarray(jax.image.resize(obs.segmentation, (int(obs.segmentation.shape[0] * scale), 
                        int(obs.segmentation.shape[1] * scale)), 'nearest'))
    rgb = np.asarray(jax.image.resize(obs.rgb, (int(obs.rgb.shape[0] * scale), 
                        int(obs.rgb.shape[1] * scale), 3), 'nearest'))
    gt_image = np.asarray(b.unproject_depth_jit(depth, intrinsics))

    return gt_image, depth, seg, rgb, intrinsics, cam_pose

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
    
    gt_images = b.unproject_depth_vmap_jit(np.stack(depths), intrinsics)

    return gt_images, depths, segs, rgbs, intrinsics

def fake_masker(point_cloud_image, segmentation, object_mask, object_ids, id, iter,size_thresh):
    object_ids = object_ids.at[iter].set(-1)
    return object_ids, object_mask

def inner_add_mask(object_mask, object_ids, segmentation, id,iter):
    object_mask += (segmentation == id)
    object_ids = object_ids.at[iter].set(id)
    return object_ids, object_mask 

def inner_fake(object_mask, object_ids, segmentation, id,iter):
    object_ids = object_ids.at[iter].set(-1)
    return object_ids, object_mask

def masker_f(point_cloud_image, segmentation, object_mask, object_ids, id, iter,size_thresh):

    mask = segmentation == id
    # Use the mask to select elements, keeping the original shape
    masked_point_cloud_segment = jnp.where(mask[..., None], point_cloud_image, jnp.nan)

    bbox_dims = custom_aabb(masked_point_cloud_segment)
    not_object = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
                    (bbox_dims[0] < 0.1),
                    (bbox_dims[1] < 0.1)),
                (jnp.greater(bbox_dims[1] + bbox_dims[0],size_thresh))),
            False),
        (bbox_dims[2] > 1.1)
    )
    return jax.lax.cond(not_object, inner_fake, inner_add_mask,*(object_mask, object_ids, segmentation, id, iter))

def custom_aabb(object_points):
    maxs = jnp.nanmax(object_points, axis = (0,1))
    mins = jnp.nanmin(object_points, axis = (0,1))
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims

@jax.jit
def get_object_mask(point_cloud_image, segmentation, size_thresh):
    segmentation_ids = jnp.unique(segmentation, size = 10, fill_value = -1)
    object_mask = jnp.zeros(point_cloud_image.shape[:2])
    object_ids = jnp.zeros(10)
    def scan_fn(carry, id):
        object_mask, object_ids, iter = carry
        object_ids, object_mask = jax.lax.cond(id == -1, fake_masker, masker_f,*(point_cloud_image, segmentation, object_mask, object_ids, id, iter,size_thresh))
        return (object_mask, object_ids, iter + 1), None
    
    (object_mask, object_ids, _), _ = jax.lax.scan(scan_fn, (object_mask, object_ids, 0), segmentation_ids)
                                               
    object_mask = object_mask > 0
    return object_ids, object_mask




WALL_Z = 14.
CAM_POSE = np.array([[ 1,0,0,0],
[0,0,-1,-4.5], # 4.5 is an arbitrary value
[ 0,1,0,1.5],
[ 0,0,0,1]])
World2Cam = np.linalg.inv(CAM_POSE)
World2Cam[1:3] *= -1
CAM_POSE_CV2 = np.linalg.inv(World2Cam)
cam_pose = CAM_POSE_CV2

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

@jax.jit
def splice_image_new(rendered_object_image, obs_image_complement, far=150.0):
    keep_masks = jnp.logical_or(
        jnp.logical_and((rendered_object_image[...,2] <= obs_image_complement[..., 2]) * 
        rendered_object_image[...,2] > 0.0, (obs_image_complement[...,2] >= far))
        ,
        (obs_image_complement[...,2] == 0)
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images

splice_image_vmap = jax.vmap(splice_image, in_axes = (0,None))
splice_image_double_vmap = jax.vmap(splice_image, in_axes = (0,None))



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

OBJECT MASKING is wrong for frames where object is entering from the top
"""

def preprocess_mcs_physics_scene(observations, MIN_DIST_THRESH = 0.6, scale = 0.1):
    # preprocess object information before running it through model by using the gt_images
    T = len(observations)
    if T < 120: # gravity hack
        size_thresh = 10
        is_gravity = True
    else:
        size_thresh = 2.4
        is_gravity = False
    review_stack = []
    init_queue = []
    review_id = 0
    registered_objects = []
    gt_images = []
    gt_images_bg = []
    gt_images_obj = []
    # Rule, for first frame, process any object as object model
    # Afterwards, everything must come from the edges

    get_distance = lambda x,y : np.linalg.norm(x[:3,3]-y[:3,3])
    print("Extracting Meshes")
    obj_pixels = []

    poses = [[] for _ in range(T)]

    for t in tqdm(range(T)):
    # for t in range(T):
        gt_image, depth, seg, rgb, intrinsics, cam_pose = observations_to_data_by_frame(observations, t, scale = 1)
        gt_image = np.asarray(gt_image)
        gt_images.append(gt_image)
        # print("t = ",t)
        seg_ids = np.unique(seg)
        obj_ids_fixed_shape, obj_mask = get_object_mask(gt_image, seg, size_thresh)
        # remove all the -1 indices
        obj_ids = np.delete(np.sort(np.unique(obj_ids_fixed_shape)),0) # This will not be jittable
        # print(obj_ids)
        depth_bg = depth * (1.0 - obj_mask) + intrinsics.far * (obj_mask)
        depth_obj = depth * (obj_mask) + intrinsics.far * (1.0 - obj_mask)
        gt_images_bg.append(np.asarray(b.t3d.unproject_depth(depth_bg, intrinsics)))
        gt_images_obj.append(np.asarray(b.t3d.unproject_depth(depth_obj, intrinsics)))
        obj_pixel_ct = 0

        for obj_id in obj_ids:

            num_pixels = np.sum(seg == obj_id)
            obj_pixel_ct += num_pixels
            point_cloud_segment = gt_image[seg == obj_id]
            dims, pose = b.utils.aabb(point_cloud_segment)
            poses[t].append(pose)
            rows, cols = np.where(seg == obj_id)
            distance_to_edge_1 = min(np.abs(rows - 0).min(), np.abs(rows - intrinsics.height + 1).min())
            distance_to_edge_2 = min(np.abs(cols - 0).min(), np.abs(cols - intrinsics.width + 1).min())
            # print(distance_to_edge_1, distance_to_edge_2)

            if t == 0:
                init_object_model = True
                init_object_model_metadata = {
                    't_init' : t,
                    'pose' : pose,
                    't_full' : t,
                    'full_pose' : pose,
                    'num_pixels' : num_pixels,
                    'mask' : seg == obj_id
                }
                # init_object_model = False
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
                            't_init' : t,
                            'distance_to_edge_1' : distance_to_edge_1,
                            'distance_to_edge_2' : distance_to_edge_2,
                            'updating_pose' : pose,
                            'init_pose' : pose
                        }
                    )
            if len(review_stack) > 0:
                # find which object under review is the closest
                distances_rs = [get_distance(pose,r['updating_pose']) for r in review_stack if r['t_init'] == t - 1]

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
                            't_init' : init_queue[init_queue_idx]['t_init'],
                            'pose' : init_queue[init_queue_idx]['init_pose'],  # TODO A MORE ACCURATE ESTIMATION OF INIT POSE
                            'full_pose' : pose,
                            't_full' : t,
                            'num_pixels': num_pixels
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
                voxelized = np.rint(point_cloud_segment / resolution).astype(np.int32)
                min_z = voxelized[:,2].min()
                depth_val = voxelized[:,2].max() - voxelized[:,2].min()

                front_face = voxelized[voxelized[:,2] <= min_z+20, :]
                slices = [front_face]
                for i in range(depth_val):
                    slices.append(front_face + np.array([0.0, 0.0, i]))
                full_shape = np.vstack(slices) * resolution

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
                print("Adding new mesh for t = {}",init_object_model_metadata['t_init'])
        # Ensure the every review in the review stack has the same time step as the current review
        del_idxs = []
        for i in list(reversed(range(len(review_stack)))):
            if review_stack[i]['t_init'] < t:
                print("Review Stack not resolved, object may have left view, deleting review")
                del review_stack[i]

        if len(review_stack) > 0 and (not np.all([r['t_init'] == t for r in review_stack])):
            print("REVIEW STACK HAS NOT BEEN FULLY RESOLVED, LOGIC ERROR")

        obj_pixels.append(obj_pixel_ct)

    gt_images = np.stack(gt_images)
    gt_images_bg = np.stack(gt_images_bg)
    gt_images_obj = np.stack(gt_images_obj)

    # now extract the data at low resolution
    print("Extracting downsampled data")
    gt_images_downsampled, _, _, _, intrinsics_downsampled = observations_to_data(observations,scale = scale)
    # get new height and width
    new_h = intrinsics_downsampled.height
    new_w = intrinsics_downsampled.width
    # resize the depths
    depths_bg = [jax.image.resize(x[...,2], (new_h, new_w), 'nearest') for x in gt_images_bg]
    depths_obj = [jax.image.resize(x[...,2], (new_h, new_w), 'nearest') for x in gt_images_obj]
    # get new point clouds based on new intrisics
    gt_images_bg_downsampled = jnp.stack([b.t3d.unproject_depth(x, intrinsics_downsampled) for x in depths_bg])
    gt_images_obj_downsampled = jnp.stack([b.t3d.unproject_depth(x, intrinsics_downsampled) for x in depths_obj])

    # # hack to determine gravity scene
    # is_gravity = len(init_queue) > 0


    return (gt_images_downsampled,gt_images_bg_downsampled,gt_images_obj_downsampled,intrinsics_downsampled),(gt_images, gt_images_bg, gt_images_obj,intrinsics), registered_objects, obj_pixels, is_gravity, poses, cam_pose


def multiview(gt_images, gt_images_bg, gt_images_obj, tr,t):
    return b.multi_panel([b.scale_image(b.get_depth_image(gt_images_obj[t][...,2]),4),
                b.scale_image(b.get_depth_image(gt_images_bg[t][...,2]),4),
                b.scale_image(b.get_depth_image(gt_images[t][...,2]),4),
                b.scale_image(b.get_depth_image(tr.get_retval()[0][t][...,2]),4),
                b.scale_image(b.get_depth_image(tr.get_retval()[1][t][...,2]),4)
                ],labels = ['obj', 'bg', 'gt/sampled', 'rendered', 'rendered_obj'])

def multiview_video(gt_images, gt_images_bg, gt_images_obj, tr, scale = 3, framerate = 30):
    T = gt_images.shape[0]
    images = []
    for t in range(T):
        images.append(b.multi_panel([b.scale_image(b.get_depth_image(gt_images_obj[t][...,2]),scale),
                    b.scale_image(b.get_depth_image(gt_images_bg[t][...,2]),scale),
                    b.scale_image(b.get_depth_image(gt_images[t][...,2]),scale),
                    b.scale_image(b.get_depth_image(tr.get_retval()[0][t][...,2]),scale),
                    b.scale_image(b.get_depth_image(tr.get_retval()[1][t][...,2]),scale)
                    ],labels = ['obj', 'bg', 'gt/sampled', 'rendered', 'rendered_obj']))
    return display_video(images, framerate=framerate)
    

def separating_axis_test(axis, box1, box2):
    """
    Projects both boxes onto the given axis and checks for overlap.
    """
    min1, max1 = project_box(axis, box1)
    min2, max2 = project_box(axis, box2)

    return jax.lax.cond(jnp.logical_or(max1 < min2, max2 < min1), lambda: False, lambda: True)

    # if max1 < min2 or max2 < min1:
    #     return False
    # return True

def project_box(axis, box):
    """
    Projects a box onto an axis and returns the min and max projection values.
    """
    corners = get_transformed_box_corners(box)
    projections = jnp.array([jnp.dot(corner, axis) for corner in corners])
    return jnp.min(projections), jnp.max(projections)

def get_transformed_box_corners(box):
    """
    Returns the 8 corners of the box based on its dimensions and pose.
    """
    dim, pose = box
    corners = []
    for dx in [-dim[0]/2, dim[0]/2]:
        for dy in [-dim[1]/2, dim[1]/2]:
            for dz in [-dim[2]/2, dim[2]/2]:
                corner = jnp.array([dx, dy, dz, 1])
                transformed_corner = pose @ corner
                corners.append(transformed_corner[:3])
    return corners

def are_bboxes_intersecting(dim1, dim2, pose1, pose2):
    """
    Checks if two oriented bounding boxes (OBBs), which are AABBs with poses, are intersecting using the Separating 
    Axis Theorem (SAT).

    Args:
        dim1 (jnp.ndarray): Bounding box dimensions of first object. Shape (3,)
        dim2 (jnp.ndarray): Bounding box dimensions of second object. Shape (3,)
        pose1 (jnp.ndarray): Pose of first object. Shape (4,4)
        pose2 (jnp.ndarray): Pose of second object. Shape (4,4)
    Output:
        Bool: Returns true if bboxes intersect
    """
    box1 = (dim1, pose1)
    box2 = (dim2, pose2)

    # Axes to test - the face normals of each box
    axes_to_test = []
    for i in range(3):  # Add the face normals of box1
        axes_to_test.append(pose1[:3, i])
    for i in range(3):  # Add the face normals of box2
        axes_to_test.append(pose2[:3, i])

    # Perform SAT on each axis
    count_ = 0
    for axis in axes_to_test:
        count_+= jax.lax.cond(separating_axis_test(axis, box1, box2), lambda:0,lambda:-1)

    return jax.lax.cond(count_ < 0, lambda:False,lambda:True)

are_bboxes_intersecting_jit = jax.jit(are_bboxes_intersecting)
# For one reference pose (object 1) and many possible poses for the second object
are_bboxes_intersecting_many = jax.vmap(are_bboxes_intersecting, in_axes = (None, None, None, 0))
are_bboxes_intersecting_many_jit = jax.jit(are_bboxes_intersecting_many)


def get_particle_images(intrinsics_orig, inferred_poses, T, cam_pose = CAM_POSE_CV2, radius = 0.1):
    viz = o3dviz.Open3DVisualizer(intrinsics_orig)
    viz_poses = jnp.reshape(inferred_poses, (inferred_poses.shape[0], 
                inferred_poses.shape[1]*inferred_poses.shape[2],
                inferred_poses.shape[3], inferred_poses.shape[4]))
    viz_poses_world = jnp.einsum("ij,abjk->abik",cam_pose, viz_poses)
    particle_images = []
    for t in tqdm(range(T)):
        pcd = viz.make_particles(viz_poses_world[t,:,:3,3], radius = radius)
        particle_arr = viz.capture_image(intrinsics_orig,cam_pose).rgb
        particle_img = Image.fromarray(np.array(particle_arr).astype(np.uint8)[...,:3])
        particle_images.append(particle_img)
        viz.clear()
    del viz
    return particle_images

def plot_vector(ax, origin, direction, color):
    """Plot a vector from 'origin' in the 'direction' with a specific 'color'."""
    ax.quiver(*origin, *direction, color=color, arrow_length_ratio=0.1)

def visualize_rotation_matrices(rot_matrices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Creating a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", alpha=0.1)

    # Choose a unit vector (e.g., along the z-axis)
    val = 1/(3)**(0.5)
    unit_vector = np.array([val, -val, val])

    # Color for the vector
    color = "r"  # Red
    plot_vector(ax, [0, 0, 0], unit_vector, "b")
    # For each rotation matrix
    for R in rot_matrices:
        # Transform the unit vector by the rotation matrix
        transformed_vector = R @ unit_vector
        plot_vector(ax, [0, 0, 0], transformed_vector, color)

    # Setting the limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

def gravity_scene_plausible(poses, intrinsics, cam_pose, observations):

    # account for case where support is identified as an object
    max_poses_detected = 0
    for i in range(len(poses)):
        if len(poses[i]) > max_poses_detected:
            max_poses_detected = len(poses[i])
    
    if max_poses_detected > 2:
        new_poses = [[] for _ in range(len(poses))]
        support_pose = poses[0][0]
        for i in range(len(poses)):
            support_j = None
            min_dist = np.inf
            for j in range(len(poses[i])):
                diff = np.linalg.norm(support_pose[:3,3] - poses[i][j][:3,3])
                if diff < min_dist:
                    min_dist = diff
                    support_j = j
            support_pose = poses[i][support_j]
            for j in range(len(poses[i])):
                if j != support_j:
                    new_poses[i].append(poses[i][j])

        poses = new_poses

    # for i,x in enumerate(poses):
    #     print(i,len(x))


    idx = 0
    # while len(poses[idx]) < 2:
    #     idx += 1

    while True:
        distances = []
        for next_pose in poses[idx+1]:
            for cur_pose in poses[idx]:
                distances.append(np.linalg.norm(next_pose[:3,3] - cur_pose[:3,3]))
        if (len([True for x in distances if x == 0.0]) == 2 and len(distances) == 4) or (0.0 in distances and len(distances)==1):
            break
        idx += 1


    correct_id = 0
    min_height = np.inf
    for obj_id in range(len(poses[idx])):
        if (cam_pose @ poses[idx][obj_id])[2,3] < min_height:
            min_height = (cam_pose @ poses[idx][obj_id])[2,3]
            correct_id = obj_id

    ref_pose = poses[idx][correct_id]
    # print(idx)
    # print((cam_pose @ ref_pose)[:3,3])

    gt_image, depth, seg, *_= observations_to_data_by_frame(observations, idx, scale = 1)
    obj_seg = None
    unique_ids = np.unique(seg)
    min_dist = np.inf
    for u_id in unique_ids:
        point_cloud_segment = gt_image[seg == u_id]
        _, pose = b.utils.aabb(point_cloud_segment)
        dist = np.linalg.norm(pose[:3,3] - ref_pose[:3,3])
        if dist < min_dist:
            min_dist = dist
            obj_seg = u_id

    if max_poses_detected > 2:
        ref_depth_obj = np.where(seg != obj_seg, intrinsics.far, depth)
        ref_depth_bg  = np.where(seg != obj_seg, depth, intrinsics.far)

    obj_indices = np.argwhere(ref_depth_obj != intrinsics.far)
    bottom_i = np.max(obj_indices[:,0])

    base_pixel_offset = 20
    base_depth_delta_thresh = 1
    line = ref_depth_bg[bottom_i+base_pixel_offset]
    base_j_min = None
    base_j_max = None
    on_support = False
    found_base = False
    for j in range(len(line)-1):
        if not on_support and line[j+1] < line[j] - base_depth_delta_thresh:
            base_j_min = j
            on_support = True
        if on_support and line[j+1] > line[j] + base_depth_delta_thresh:
            base_j_max = j
            on_support = False
            found_base = True
        if j == len(line) - 2 and not found_base:
            print("Error: There is no base for support")
            return True, [True for _ in range(len(poses))], 1e+20
                        
    pixels_stable = np.sum(np.logical_and(obj_indices[:,1] <= base_j_max , obj_indices[:,1] > base_j_min))
    pixels_unstable = np.sum(np.logical_or(obj_indices[:,1] > base_j_max , obj_indices[:,1] <= base_j_min))
    stable = pixels_stable >= pixels_unstable
    ref_height = (cam_pose @ ref_pose)[2,3]
    end_height = (cam_pose @ poses[-1][0])[2,3]
    # fell = ref_height > end_height + 0.2
    fell = np.linalg.norm(ref_pose[:3,3] - poses[-1][0][:3,3]) > 0.1
    perc = 100*(np.abs(pixels_unstable - pixels_stable)/(pixels_stable+pixels_unstable))

    print(f"Unstable: {pixels_unstable}")
    print(f"Stable: {pixels_stable}")
    print(f"Fell? ", fell)
    print(f"Perc diff: {perc}")

    # if perc < 5: # any outcome should be plausible
    #     return True, [True for _ in range(len(poses))], 1e+20

    t_violation = 1e+20
    if stable and fell:
        plausible = False
        t_violation = idx
    elif not stable and not fell:
        plausible = False
        t_violation = idx
    else:
        plausible = True

    plausibility_list = []
    for t in range(len(poses)):
        if t >= t_violation:
            plausibility_list.append(False)
        else:
            plausibility_list.append(True)

            
    return plausible, plausibility_list, t_violation

def threedp3_likelihood_arijit(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = (distances < variance/2) / variance
    average_probability = 1 * probabilities_per_pixel.mean()
    return average_probability

threedp3_likelihood_arijit_vmap = jax.vmap(threedp3_likelihood_arijit, in_axes=(None,0,None,None))
threedp3_likelihood_arijit_double_vmap = jax.vmap(threedp3_likelihood_arijit, in_axes=(0,0,None,None))

def outlier_gaussian(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = jax.scipy.stats.norm.pdf(
        distances,
        loc=0.0, 
        scale=variance
    )
    average_probability = 0.01 * probabilities_per_pixel.sum()
    return average_probability

def determine_shape_constancy_plausibility(num_objects, last_gt_image, last_gt_image_bg, last_rend):
    # if not plausibility:
    #     return 0
    best_p_index = np.argmax([outlier_gaussian(last_gt_image, last_rend[i], 0.5, None) for i in range(3)])
    # best_p_index = np.argmax([threedp3_likelihood_arijit(last_gt_image, last_rend[i], 0.5, None) for i in range(3)])
    if num_objects == 2:
        # var = 0.108
        var = 1.0621818181818181
        xx1 = threedp3_likelihood_arijit(last_gt_image,last_gt_image,var,None)
        xx2 = threedp3_likelihood_arijit(last_gt_image,last_rend[best_p_index],var,None)
        xx3 = threedp3_likelihood_arijit(last_gt_image,last_gt_image_bg,var,None)
        frac = (xx2-xx3)/(xx1-xx3)
        print(f"{num_objects} objects, frac : {frac}")
        if frac < 0.7291686:
        # if frac < 0.38143346:
            return 0
        else:
            return 1
    elif num_objects == 1:
        var = 1.415
        # var = 0.115
        xx1 = threedp3_likelihood_arijit(last_gt_image,last_gt_image,var,None)
        xx2 = threedp3_likelihood_arijit(last_gt_image,last_rend[best_p_index],var,None)
        xx3 = threedp3_likelihood_arijit(last_gt_image,last_gt_image_bg,var,None)
        frac = (xx2-xx3)/(xx1-xx3)
        # if frac < 0.4827912:
        print(f"{num_objects} objects, frac : {frac}")
        if frac < 0.74286604:
            return 0
        else:
            return 1
    else:
        return np.random.choice([0,1])
    
def determine_plausibility(results, offset = 3, rend_fraction_thresh = 0.75):
    # check to see if object is falling from top

    T = results['resampled_indices'].shape[0] - offset
    tsteps_before_start = results['inferred_poses'].shape[0] - T

    height, width = results['intrinsics'].height, results['intrinsics'].width
    starting_indices = results['all_obj_indices'][tsteps_before_start - offset]
    if starting_indices is not []:
        mean_i, mean_j = np.median(starting_indices[:,0]), np.median(starting_indices[:,1])
        from_top = (mean_i < height/2) and (mean_j > mean_i) and (mean_j < width -mean_i)
    else:
        from_top = False

    # first get base indices to reflect resampled particles
    n_particles = results['resampled_indices'].shape[1]
    resample_bools = np.all(results['resampled_indices'] == np.arange(n_particles), axis = 1)
    base_indices = np.arange(n_particles)
    for i in range(results['resampled_indices'].shape[0]):
        base_indices = base_indices[results['resampled_indices'][i]]
    # then get the rendering scores based on resampled_indices
    rend = np.array(results["rend_ll"][offset:,base_indices])
    # get the worst rendered scores (object-less)
    WR = results["worst_rend"][offset:]
    # flatten rend to get the best vector across time
    rend = np.max(rend, axis = 1)

    max_rend_possible = height * width * jax.scipy.stats.norm.pdf(
        0.,
        loc=0.0, 
        scale=results["variance"]
    ) * 0.01

    t_violation = None
    plausibility_list = [True for _ in range(tsteps_before_start)]
    plausible = True
    count_violation = 0
    for t in range(T):
        if WR[t] > rend[t]:
            plausible = False
            if t_violation is None:
                t_violation = tsteps_before_start + t
        if WR[t] < max_rend_possible and WR[t] == rend[t]:
            count_violation +=1
            if t_violation is None:
                t_violation = tsteps_before_start + t
        # if from_top and rend[t] > WR[t] and WR[t] >= WR[t-1] and t > T/2 and results['inferred_poses'].shape[0] < 220: # CONSIDER REMOVING THIS HACK
        #     WR_gap = max_rend_possible - WR[t]
        #     rend_gap = max_rend_possible - rend[t]
        #     rend_likelihood_fraction = (WR_gap - rend_gap)/WR_gap
        #     if rend_likelihood_fraction < rend_fraction_thresh:
        #         plausible = False
        #         if t_violation is None:
        #             t_violation = tsteps_before_start + t
        plausibility_list.append(plausible)

    if from_top and results['inferred_poses'].shape[0] < 220:
        num_objects = results['inferred_poses'].shape[2]
        plausible = determine_shape_constancy_plausibility(num_objects, results['gt_images'][-1], results['gt_images_bg'][-1], results['rendered'][-1])
        t_violation = 100
        plausibility_list = [t < t_violation for t in range(T)]


    if count_violation > 3:
        plausible = False
        plausibility_list = [t < t_violation for t in range(T)]
        
    return plausible, t_violation, plausibility_list, from_top


