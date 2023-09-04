import bayes3d as b
import jax.numpy as jnp
import jax
import os
import matplotlib.pyplot as plt
import matplotlib

b.setup_renderer


intrinsics = b.Intrinsics(
    height=1000,
    width=1000,
    fx=500.0, fy=500.0,
    cx=500.0, cy=500.0,
    near=0.01, far=10.0
)

b.setup_renderer(intrinsics)
model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
meshes = []
for idx in range(1,22):
    mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
    b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=10.0/1000.0)

b.ycb_loader.MODEL_NAMES[11]

IDXs = 21 #10 #11
frames = 40
devices = 1
dots = 250

lifetime = 5 #keep 1-1/5 of the dots after every frame update
point_rad = 5

SAMPLES = 500


axis_orientations = jax.random.uniform(jax.random.PRNGKey(11), shape=(3,SAMPLES))

import itertools

for IDX, s in itertools.product(range(0,IDXs), range(SAMPLES)):
#for IDX, s in itertools.product(range(1,3), range(SAMPLES)):

    axis = axis_orientations[:,s]

    pc = jnp.array(b.RENDERER.meshes[IDX].vertices)
    pc_subsample_start = pc[jax.random.choice(jax.random.PRNGKey(10), jnp.arange(pc.shape[0]), shape=(dots,) )] #want 1000 dots total
    pc_replacements = pc[jax.random.choice(jax.random.PRNGKey(0), jnp.arange(pc.shape[0]), shape=(frames,dots//lifetime) )]

    pc_subsamples = jnp.zeros((frames,*pc_subsample_start.shape))
    pc_subsamples = pc_subsamples.at[0,...].set(pc_subsample_start)
    for i in range(1,frames):
        pc_subsamples = pc_subsamples.at[i,...].set(pc_subsamples[i-1,...])
        sampled_indices = jax.random.choice(jax.random.PRNGKey(i), jnp.arange(dots), shape=(dots//lifetime,) )
        pc_subsamples = pc_subsamples.at[i,sampled_indices,...].set(pc_replacements[i,...])


    poses = jnp.array([b.t3d.inverse_pose(b.t3d.transform_from_pos_target_up(
            jnp.array([0.0, 2.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
    )) @ b.t3d.transform_from_axis_angle(axis, angle) for angle in jnp.linspace(0.0, 4*jnp.pi, frames)])


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

    def render_point_light(pose, pc_to_render, key):
        pc_in_camera_frame = b.t3d.apply_transform(pc_to_render, pose)
        img = b.render_point_cloud(pc_in_camera_frame, intrinsics)
        rendered_image = point_cloud_img = b.RENDERER.render_single_object(pose,  jnp.int32(IDX))[:,:,:3]
        mask = (rendered_image[:,:,2] < intrinsics.far)
        
        matches = (jnp.abs(img[:,:,2] - rendered_image[:,:,2]) < 0.05)
        
        flips = (jax.random.uniform(key,shape=matches.shape) < 0.0005)
        
        final_no_noise = circles(mask * matches,point_rad)
        final_with_noise = circles(mask * matches + (1.0 - mask) * flips, point_rad)

        return final_no_noise, final_with_noise


    gpus = jax.devices('gpu')
    render_point_light_parallel_jit = jax.jit(jax.vmap(render_point_light, in_axes=(0,0, 0)))



    key = jax.random.PRNGKey(100)
    keys = jax.random.split(jax.random.PRNGKey(100), poses.shape[0])
    images_no_noise, images = render_point_light_parallel_jit(poses, pc_subsamples, keys)

    stim_path = './stimuli/obj'+str(IDX).zfill(2)
    stim_gt_path = './stimuli_gt/obj'+str(IDX).zfill(2)

    if not os.path.exists(stim_path):
        os.mkdir(stim_path)

    if not os.path.exists(stim_gt_path):
        os.mkdir(stim_gt_path)
        #with open(stim_gt_path+"/name.txt", "w") as text_file:
        #    text_file.write(b.ycb_loader.MODEL_NAMES[IDX])
        

    viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in images ]
    b.make_gif_from_pil_images(viz, stim_path+"/vec"+str(s).zfill(3)+".gif")
    # viz[0].save('out_frame_m.png')
    # viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in images_no_noise ]
    # b.make_gif_from_pil_images(viz, "./stimuli/obj"+str(IDX).zfill(2)+"vec"+str(s).zfill(3)+"out_clean.gif")

    static = jnp.repeat(images[0,...][jnp.newaxis,...], frames, axis=0)
    viz = [b.get_depth_image(1.0 - point_light_image * 1.0, cmap=matplotlib.colormaps['Greys']) for point_light_image in jnp.concatenate((static, images_no_noise, images),axis=2)]
    b.make_gif_from_pil_images(viz, stim_gt_path+"/vec"+str(s).zfill(3)+".gif")