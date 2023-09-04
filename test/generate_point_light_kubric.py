# %%
import jax.numpy as jnp
import bayes3d as b
import trimesh
import os
import numpy as np
import trimesh
from tqdm import tqdm
from bayes3d._rendering.photorealistic_renderers.kubric_interface import render_many
import png2avi as p2a
import jax
import matplotlib
# import matplotlib.pyplot as plt


# %%
scene_ind = 57 #54
im_ind = 1

# --- creating the ycb dir from the working directory
bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img(str(scene_ind), str(im_ind), bop_ycb_dir)

# %%
intrinsics = b.Intrinsics(
    rgbd.intrinsics.height, rgbd.intrinsics.width,
    rgbd.intrinsics.fx, rgbd.intrinsics.fx,
    rgbd.intrinsics.width/2, rgbd.intrinsics.height/2,
    rgbd.intrinsics.near, 10.0 #rgbd.intrinsics.far
)


b.setup_renderer(intrinsics)

# %%
mesh_paths = []
offset_poses = []
heights = []
names = []
model_dir = os.path.join(b.utils.get_assets_dir(), "ycb_video_models/models")
for i in tqdm(gt_ids):
    mesh_path = os.path.join(model_dir, b.utils.ycb_loader.MODEL_NAMES[i],"textured.obj")
    m, pose = b.utils.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)
    m = trimesh.load(mesh_path)
    bbox, _ = b.utils.aabb(m.vertices)
    heights.append(bbox[2]) # get z axis
    names.append(b.utils.ycb_loader.MODEL_NAMES[i])
    offset_poses.append(pose)
    b.RENDERER.add_mesh_from_file(mesh_path, center_mesh=False)
    mesh_paths.append(
        mesh_path
    )

# %%
# print heights
print('Object Heights:')
for i, name in enumerate(names):
    print(name + ': ' + str(heights[i]))

# %%
poses = []
for i in range(len(gt_ids)):
    poses.append(
        gt_poses[i] @ b.t3d.inverse_pose(offset_poses[i])
    )
poses = jnp.array(poses)

# %%
# Note: the hardcoded object has to be upright - how to detect this automatically?
centered_item = 0 # hardcoded to be a number

center_obj_basis = gt_poses[centered_item]

obj_poses = jnp.einsum('jk,ikl->ijl', b.t3d.inverse_pose(center_obj_basis),poses)

# %%
scene_pc = []

for i in range(len(b.RENDERER.meshes)):
    scene_pc.append(b.t3d.apply_transform(b.RENDERER.meshes[i].vertices, obj_poses[i]))

scene_pc = np.concatenate(scene_pc)

bbox, center = b.utils.aabb(scene_pc)
minz = center[2,3]-bbox[2]/2
scene_pc_shift = b.t3d.apply_transform(scene_pc, b.t3d.inverse_pose(b.t3d.transform_from_pos(jnp.array([0,0,minz]))))

# %%
table_mesh = b.utils.make_cuboid_mesh([1,1,0.01])

max_edge = min(b.utils.aabb(table_mesh.vertices)[0])

b.RENDERER.add_mesh(trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(table_mesh.vertices, table_mesh.faces, max_edge))) # need to remesh to proper scale
frames = 20

dome_pose = np.eye(4)

vids = 1 #10

# %%
## Add scaled wood block for photorealistic kubric render
mesh_path = os.path.join(model_dir, b.utils.ycb_loader.MODEL_NAMES[15],"textured.obj")
mesh_paths.append(mesh_path)
b.RENDERER.add_mesh_from_file(mesh_path, center_mesh=True)

flat_wood_ind = len(b.RENDERER.meshes) - 1
bbox, _ = b.utils.aabb(b.RENDERER.meshes[flat_wood_ind].vertices)

tabletop_scaling_factor = [15.0, 15.0, 1.0/10]
b.RENDERER.meshes[flat_wood_ind].vertices = b.RENDERER.meshes[flat_wood_ind].vertices @ np.diag(tabletop_scaling_factor)


# %%
translated_obj_poses = []
for i in range(len(obj_poses)):
    translated_obj_poses.append(b.t3d.inverse_pose(b.t3d.transform_from_pos(jnp.array([0,0,minz]))) @ obj_poses[i])

translated_obj_poses.append(dome_pose) # wood block
translated_obj_poses.append(dome_pose) # cuboid mesh

translated_obj_poses = jnp.array(translated_obj_poses)

# %%
## Utility functions

def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset_min, rng_linear):
    while True:
        point = rng_linear.uniform(inner_radius, outer_radius, 3)
        if np.linalg.norm(point) > inner_radius and np.linalg.norm(point) < outer_radius and point[2] > z_offset_min:
            return point

# %%
# camera view rng is FIXED TRAJECTORY in this function
def get_linear_camera_motion(
    movement_speed: float,
    inner_radius: float = 0.5,
    outer_radius: float = 1,
    z_offset_min: float = 0.1,
    z_offset_max: float = 1,
    frames = 10,
    rng_linear = np.random.RandomState(12345)
):
    """Sample a linear path which starts and ends within a half-sphere shell."""

    while True:
        camera_start = np.array(sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset_min, rng_linear))
        direction = rng_linear.rand(3) - 0.5
        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement

        #check values
        print('camera start: '+str(camera_start))
        print('camera end: ' + str(camera_end))

        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
            camera_end[2] > z_offset_min and camera_end[2] < z_offset_max):
            break
    
    camera_positions = []

    for frame in range(frames):
        interp = (frame * 1.0) / frames
        pos_interp = camera_start + interp*(camera_end - camera_start)
        camera_positions.append(pos_interp)
    
    return camera_positions


# camera view rng is FIXED TRAJECTORY in this function
def get_spherical_camera_motion(
    movement_speed: float,
    inner_radius: float = 0.75,
    outer_radius: float = 1.5,
    z_offset_min: float = 0.1,
    z_offset_max: float = 1,
    frames = 10,
    rng_sphere = np.random.RandomState(12345)
):
    """Sample a spherical path which starts and ends within a half-sphere shell."""

    while True:
        camera_start = np.array(sample_point_in_half_sphere_shell(inner_radius, outer_radius, z_offset_min, rng_sphere))
        

        # movement speed is defined as unit time for the whole trajectory
        # frame interpolation is done in next block
        length = movement_speed * 1.0 

        lambda1, phi1, r = cart2sph(*camera_start)
        angle = length/r

        # Great Circle Formula:
        # angle = arrcos(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(del_lambda))
        phi2 = rng_sphere.uniform(0, np.pi/2)
        del_lambda = np.arccos((np.cos(angle) - np.sin(phi1)*np.sin(phi2))/(np.cos(phi1)*np.cos(phi2)))
        #lambda2 = rng_sphere.choice([lambda1+del_lambda, lambda1-del_lambda])
        lambda2 = lambda1+del_lambda#, lambda1-del_lambda

        camera_end = np.array(sph2cart(lambda2, phi2, r))

        #check values
        print('camera start spherical: '+str(cart2sph(*camera_start)))
        print('camera end spherical: ' + str(cart2sph(*camera_end)))

        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
            camera_end[2] > z_offset_min and camera_end[2] < z_offset_max):
            # return camera_start, camera_end
            break
    
    camera_positions = []

    for frame in range(frames):
        # linear interpolate lambda and phi angles, arc length not preserved
        interp = (frame * 1.0) / frames

        lambda1, phi1, r = cart2sph(*camera_start)
        lambda2, phi2, r = cart2sph(*camera_end)

        # angles are hacks
        phi_interp = phi1 + interp*(phi2-phi1)
        lambda_interp = lambda1 + interp*(lambda2-lambda1)

        camera_positions.append(sph2cart(lambda_interp, phi_interp, r))
    
    return camera_positions

# %%
max_camera_movement = 1.5


# interpolate the camera position between these two points
# while keeping it focused on the center of chosen object

positions = []
orientations = []

rng = np.random.RandomState(2)
# look_ind = rng.choice(len(translated_obj_poses))

look_ind = np.argmax(np.array(heights))
look_point = translated_obj_poses[look_ind][0:3,3] # look at the center of the tallest object


camera_trajectory_pos = get_spherical_camera_motion(
    movement_speed=rng.uniform(low=max_camera_movement/2.0, high=max_camera_movement), # low was 0
    inner_radius=0.75,
    outer_radius=1.5, # is outer radius too limiting?
    z_offset_min=0.1,
    z_offset_max=0.4,
    frames=frames,
    rng_sphere=rng
)

# %%
camera_poses = []
up = np.array([0,0,1])

for pos in camera_trajectory_pos:
    camera_poses.append(b.t3d.transform_from_pos_target_up(np.array(pos), look_point, up))

cam_poses = jnp.array(camera_poses)

# %%
multiframe_poses = []

for c in range(len(cam_poses)):
    frame_poses = []
    for p in range(len(translated_obj_poses)):
        frame_poses.append(b.t3d.inverse_pose(cam_poses[c]) @ translated_obj_poses[p])
    multiframe_poses.append(frame_poses)

multiframe_poses = jnp.array(multiframe_poses)

# %%
# render_indices = [0,1,2,3,4,5]
# depth_im = b.RENDERER.render_many(multiframe_poses[:,render_indices,:,:], jnp.array(render_indices))

# %% [markdown]
# ### Generate Point Light Renders

# %%
scene_subsample_proportion = 25
dots = 250 #500
lifetime = 5 #keep 1-1/5 of the dots after every frame update
point_rad = 5

# Subsample dots in point cloud
choices = rng.choice(np.arange(len(scene_pc_shift)), size = len(scene_pc_shift)//scene_subsample_proportion, replace=False)
scene_pc_subsample = scene_pc_shift[choices,:]
scene_pc_table_shift = np.concatenate((scene_pc_subsample, b.RENDERER.meshes[-2].vertices), axis=0)

# Resample fraction of dots at each frame in video according to lifetime

pc = scene_pc_table_shift
pc_subsample_start = pc[jax.random.choice(jax.random.PRNGKey(10), jnp.arange(pc.shape[0]), shape=(dots,) )] #want 1000 dots total
pc_replacements = pc[jax.random.choice(jax.random.PRNGKey(0), jnp.arange(pc.shape[0]), shape=(frames,dots//lifetime) )]

pc_subsamples = jnp.zeros((frames,*pc_subsample_start.shape))
pc_subsamples = pc_subsamples.at[0,...].set(pc_subsample_start)

for i in range(1,frames):
    pc_subsamples = pc_subsamples.at[i,...].set(pc_subsamples[i-1,...])
    sampled_indices = jax.random.choice(jax.random.PRNGKey(i), jnp.arange(dots), shape=(dots//lifetime,) )
    pc_subsamples = pc_subsamples.at[i,sampled_indices,...].set(pc_replacements[i,...])

# %%
# utlity and rendering functions

def circles(flips_xy, radius):
    centers = jnp.array((flips_xy>0).nonzero(size=5000,fill_value=jnp.inf))
    x,y = jnp.meshgrid(jnp.arange(flips_xy.shape[1]),jnp.arange(flips_xy.shape[0]))
    xymesh = jnp.array([y,x])
    distances_to_keypoints = (
        jnp.linalg.norm(xymesh[:, :,:,None] - centers[:,None, None,:],
        axis=0
    ))
    index_of_nearest_keypoint = distances_to_keypoints.argmin(2)
    distance_to_nearest_keypoints = distances_to_keypoints.min(2)
    DISTANCE_THRESHOLD = radius
    valid_match_mask = (distance_to_nearest_keypoints < DISTANCE_THRESHOLD)[...,None]
    return valid_match_mask

def render_point_light(poses, cam_pose, pc_to_render, key):
    pc_in_camera_frame = b.t3d.apply_transform(pc_to_render, b.t3d.inverse_pose(cam_pose))
    img = b.render_point_cloud(pc_in_camera_frame, intrinsics)

    # idx needs to be less hacky
    rendered_image = point_cloud_img = b.RENDERER.render(poses,  jnp.array([0,1,2,3,4,5]))[:,:,:3] # this needs to be extended to multiple objects

    
    mask = (rendered_image[:,:,2] < intrinsics.far)
    
    matches = (jnp.abs(img[:,:,2] - rendered_image[:,:,2]) < 0.05)
    #matches = (jnp.abs(img[:,:,2] - rendered_image[:,:,2]) < 0.5) ???
    
    #turn down the flips to make less noisy
    flips = (jax.random.uniform(key,shape=matches.shape) < 0.0005)
    
    final_no_noise = circles(mask * matches,point_rad)
    final_with_noise = circles(mask * matches + (1.0 - mask) * flips, point_rad)

    return final_no_noise, final_with_noise



# %%
# # test block to check correctness of depth and point cloud renders
# im = b.RENDERER.render(multiframe_poses[0][0:6],  jnp.array([0,1,2,3,4,5]))[:,:,:3]
# pcim = b.render_point_cloud(b.t3d.apply_transform(pc_subsamples[0], b.t3d.inverse_pose(cam_poses[0])), intrinsics, pixel_smudge=5)
# plt.imshow(pcim[:,:,2])
# plt.colorbar()

# %%
key = jax.random.PRNGKey(100)
keys = jax.random.split(key, multiframe_poses.shape[0])

# last item in mesh list is photorealistic wood block, can't be used for point light stimulus
b.RENDERER.meshes = b.RENDERER.meshes[:6] 

render_point_light_parallel_jit = jax.jit(jax.vmap(render_point_light, in_axes=(0,0,0,0)))
images_no_noise, images = render_point_light_parallel_jit(multiframe_poses[:,:6,:,:], cam_poses, pc_subsamples, keys)

# write video to GIF

stim_path = './stimuli/scene'+str(scene_ind).zfill(2)
stim_gt_path = './stimuli_gt/scene'+str(scene_ind).zfill(2)

# if not os.path.exists(stim_path):
#     os.mkdir(stim_path)

if not os.path.exists(stim_gt_path):
    os.makedirs(stim_gt_path)
    

# viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in images ]
# b.make_gif_from_pil_images(viz, stim_path+"/vec"+str(s).zfill(3)+".gif")
# # viz[0].save('out_frame_m.png')
# # viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in images_no_noise ]
# # b.make_gif_from_pil_images(viz, "./stimuli/obj"+str(IDX).zfill(2)+"vec"+str(s).zfill(3)+"out_clean.gif")

traj = 1
fps = 8

static = jnp.repeat(images[0,...][jnp.newaxis,...], frames, axis=0)
viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in jnp.concatenate((static, images_no_noise, images),axis=2)]
b.make_gif_from_pil_images(viz, stim_gt_path+"/vec"+str(traj).zfill(3)+".gif", fps=fps)

# %%
print(mesh_paths)
len(mesh_paths)


# %% [markdown]
# ### Interface to photorealistic renderer

# %%
from bayes3d._rendering.photorealistic_renderers.kubric_interface_background import render_many

# %%
kubric_pose_idx = [0,1,2,3,4,6] # this is always the number of objects in scene plus last element in mesh list
rgbds = render_many(mesh_paths, multiframe_poses[:,kubric_pose_idx,:,:], intrinsics, tabletop_scaling_factor = tabletop_scaling_factor)

vid = 1
im_dir = "ku_scene_vids_linear_"+str(scene_ind)+"/frames"+str(vid)+"/images"

if not os.path.exists(im_dir):
    os.makedirs(im_dir)

for frame, rgbd in enumerate(rgbds):
    b.get_rgb_image(rgbd.rgb).save(im_dir+"/image{:03d}.png".format(frame))


p2a.save(image_folder = im_dir, video_name = 'ku_scene_vids_linear_'+str(scene_ind)+'/linear'+str(vid)+'.avi', fps=fps)

# write camera position and quaternions

cp = []
co = []

for c_ind in range(len(cam_poses)):
    cp.append(np.array(cam_poses[c_ind,:3,3]))
    co.append(np.array(b.t3d.rotation_matrix_to_quaternion(cam_poses[c_ind,:3,:3])))

gt_poses = np.concatenate((np.array(cp), np.array(co)),axis=1)
np.savetxt('ku_scene_vids_linear_'+str(scene_ind)+'/cam_pos_ori'+str(vid)+'.txt', gt_poses)

# %%
# ground truth poses seem to come from manual optimization, which explains why there are weird offsets in object poses

# %%



