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
    intrinsics_data = observations[0].intrinsics
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

    return gt_image, depth, seg, rgb, intrinsics

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

    bbox_dims = custom_aabb(masked_point_cloud_segment)
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
    return dims

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
[0,0,-1,-4.5], # 4.5 is an arbitrary value
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

OBJECT MASKING is wrong for frames where object is entering from the top
"""

def preprocess_mcs_physics_scene(observations, MIN_DIST_THRESH = 0.6, scale = 0.1):
    # preprocess object information before running it through model by using the gt_images
    T = len(observations)
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
        gt_image, depth, seg, rgb, intrinsics = observations_to_data_by_frame(observations, t, scale = 1)
        gt_image = np.asarray(gt_image)
        gt_images.append(gt_image)
        # print("t = ",t)
        seg_ids = np.unique(seg)
        obj_ids_fixed_shape, obj_mask = get_object_mask(gt_image, seg)
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
                    't_fully_in_scene' : t
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

    # hack to determine gravity scene
    is_gravity = len(init_queue) > 0

    return (gt_images_downsampled,gt_images_bg_downsampled,gt_images_obj_downsampled,intrinsics_downsampled),(gt_images, gt_images_bg, gt_images_obj,intrinsics), registered_objects, obj_pixels, is_gravity, poses


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

@genjax.gen
def physics_stepper_nov28(all_poses, t, t_full, i, friction, gravity):
    # TODO: SAMPLING FRICTION SCHEME --> can be of a hmm style

    #################################################################
    # First let us consider timestep t-1
    #################################################################
    # Step 2: find world pose
    pose_prev = all_poses[t-1]
    pose_prev_world = cam_pose @ pose_prev

    # Step 3: check if we are already on the floor
    bottom_z, top_z, center_to_bottom = get_height_bounds(i, pose_prev_world)
    # within 20% of the object's height in world frame
    already_on_floor = jnp.less_equal(bottom_z,0.2 * (top_z - bottom_z))
    
    # Step 1: Find world velocity
    vel_pose_camera = jnp.linalg.solve(all_poses[t-2], all_poses[t-1])
    pre_vel_xyz_world = cam_pose[:3,:3] @ vel_pose_camera[:3,3]
    mag_xyz = jnp.linalg.norm(pre_vel_xyz_world)
    
    mag_xyz_friction = jnp.linalg.norm(pre_vel_xyz_world[:2]) - friction * mag_xyz

    mag_xyz_friction = jax.lax.cond(
        jnp.less_equal(jnp.abs(mag_xyz_friction),3e-2),
        lambda:0.0,
        lambda:mag_xyz_friction)
    
    # jax.lax.cond(t >= t_full,
    #              lambda: jprint("mag : {}",mag_xyz_friction),
    #              lambda: None)
    
    


    dir_xyz_world = get_translation_direction(all_poses, t_full, t, already_on_floor)

    # Step 5: find peturbed velocity (equal to original norm) with random rotation
    perturbed_rot_pose = GaussianVMFPoseUntraced()(jnp.eye(4), *(1e-20, 100000.0))  @ "perturb"

    # perturbed_rot_pose = jax.lax.cond(mag_xyz < 0.0,
    #                              lambda:jnp.eye(4),
    #                              lambda: perturbed_rot_pose)

    dir_xyz_world = perturbed_rot_pose[:3,:3] @ dir_xyz_world # without friction

    # Step 7: Determine mag and gravity
    mag, gravity = jax.lax.cond(already_on_floor,lambda:(mag_xyz_friction,gravity),lambda:(mag_xyz, gravity))

    # jprint("{},{}",dir_xyz_world, mag)

    vel_xyz_world = mag * dir_xyz_world
    # Step 6: apply gravity to perturbed velocity
    vel_xyz_world = vel_xyz_world.at[2].set(vel_xyz_world[2] - gravity * 1./20)

    vel_xyz_camera = inverse_cam_pose[:3,:3] @ vel_xyz_world

    # Step 8: Get velocity update in camera frame
    vel = pose_prev.at[:3,3].set(vel_xyz_camera)

    # jprint("mag : {}",mag_xyz_friction)
    # jprint("on_floor : {}",already_on_floor)

    # Step 9: Identify next pose
    next_pose = pose_prev.at[:3,3].set(pose_prev[:3,3] + vel[:3,3]) # trans only, no rot

    # Step 10: Ensure new bottom of object is above floor --> ground collision
    next_pose_world = cam_pose @ next_pose
    bottom_z,_,center_to_bottom = get_height_bounds(i, next_pose_world)
    next_pose = jax.lax.cond(
        jnp.less_equal(bottom_z,0),
        lambda:inverse_cam_pose @ next_pose_world.at[2,3].set(center_to_bottom),
        lambda:next_pose
    )

    return next_pose