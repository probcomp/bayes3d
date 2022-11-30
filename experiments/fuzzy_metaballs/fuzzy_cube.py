import os
from jax3dp3.fuzzy import fuzzy_renderer as fm_render
import jax.numpy as jnp
import jax
import jax3dp3.utils
import numpy as np
from jax3dp3.viz.img import save_depth_image,save_rgb_image, get_depth_image, multi_panel
from tqdm import tqdm
from jax3dp3.transforms_3d import transform_from_axis_angle
from jax3dp3.shape import get_cube_shape
from jax3dp3.rendering import render_planes_rays
import functools
from jax3dp3.viz.img import save_depth_image
from jax3dp3.triangle_renderer import render_triangles
from jax3dp3.transforms_3d import transform_from_pos, apply_transform
import jax3dp3.transforms_3d as t3d
import trimesh


image_size = (120,120)

hyperparams = fm_render.hyperparams
NUM_MIXTURE = 100
beta_2 = jnp.float32(np.exp(hyperparams[0]))
beta_3 = jnp.float32(np.exp(hyperparams[1]))
beta_4 = jnp.float32(np.exp(hyperparams[2]))
beta_5 = -jnp.float32(np.exp(hyperparams[3]))

def rays_from_R_T(R,T, image_size=image_size):
    height, width = image_size
    cx = (width-1)/2
    cy = (height-1)/2
    vfov_degrees = 60
    f = (height/np.tan((np.pi/180)*float(vfov_degrees)/2))*0.5
    K = np.array([[f, 0, cx],[0,f,cy],[0,0,1]])
    pixel_list = (np.array(np.meshgrid(width-np.arange(width)-1,height-np.arange(height)-1,[0]))[:,:,:,0]).reshape((3,-1)).T
    camera_rays = (pixel_list - K[:,2])/np.diag(K)
    camera_rays[:,-1] = 1

    translation = T

    camera_rays = camera_rays @ np.array(R ).T 
    trans = np.tile(translation[None],(camera_rays.shape[0],1))

    camera_starts_rays = np.stack([camera_rays,trans],1)
    return jnp.array(camera_starts_rays)

camera_angle_sweep = jnp.linspace(0.0, 2*jnp.pi, 10)
distance = 5.0

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = (10.0*mesh.vertices)[mesh.faces] * jnp.array([1.0, -1.0, 1.0])

cameras_list = []
alpha_list = []
all_depths = []

identity_rays = rays_from_R_T(jnp.eye(3), jnp.zeros(3))
for angle in camera_angle_sweep:
    R = transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), angle)[:3,:3]
    T = R @ jnp.array([0.0, 0.0, -distance])

    pose = t3d.transform_from_rot_and_pos(R,T)
    data = render_triangles(jnp.linalg.inv(pose), trimesh_shape, identity_rays[:,0,:].reshape(1,-1,3))[0]

    rays = rays_from_R_T(R,T)
    cameras_list.append(rays)

    alpha_list.append(data[:,2] != 0.0)
    all_depths.append(data[:,2])


gt_viz_images = [get_depth_image(sil.reshape(image_size)) for sil in alpha_list]
multi_panel(
    gt_viz_images,
    None,
    10,
    50,
    20
).save("all_views.png")



def objective(params,true_alpha):
    CLIP_ALPHA = 1e-6
    means,prec,weights_log,camera_rays,beta2,beta3,beta4,beta5 = params
    render_res = fm_render.render(means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5)

    est_alpha = render_res[2]
    est_alpha = jnp.clip(est_alpha,CLIP_ALPHA,1-CLIP_ALPHA)
    mask_loss = - ((true_alpha * jnp.log(est_alpha)) + (1-true_alpha)*jnp.log(1-est_alpha))
    return mask_loss.mean()
grad_render3 = jax.jit(jax.value_and_grad(objective))

all_cameras = jnp.array(cameras_list).reshape((-1,2,3))
all_sils = jnp.array(alpha_list).ravel().astype(jnp.float32)

center = jnp.array([0.0, 0.0, 0.0])
shape_scale = 2.0
rand_mean = center+np.random.multivariate_normal(mean=[0,0,0],cov=1e-1*np.identity(3)*shape_scale,size=NUM_MIXTURE)
rand_weight_log = jnp.log(np.ones(NUM_MIXTURE)/NUM_MIXTURE)
rand_sphere_size = 14
rand_prec = jnp.array([np.identity(3)*rand_sphere_size/shape_scale for _ in range(NUM_MIXTURE)])

init_t,stds,est_alpha = fm_render.render(rand_mean, rand_prec, rand_weight_log, all_cameras, beta_2, beta_3, beta_4, beta_5)

from jax.example_libraries import optimizers

# Number of optimization steps
Niter = 400
# number of images to batch gradients over

loop = tqdm(range(Niter))

# babysit learning rates
# adjust_lr = DegradeLR(1e-3,0.5,train_size//2,train_size//10,-1e-3)

opt_init, opt_update, opt_params = optimizers.adam(3e-2)
tmp = [rand_mean,rand_prec,rand_weight_log]
opt_state = opt_init(tmp)

losses = []
accum_grad = None
grad_counter = 0

# Viz
optimization_images = []
scaling_factor = 4
image_size_visualization = (image_size[0]*scaling_factor, image_size[1]*scaling_factor)

cam_idx = 5

optimization_depths = []
for i in loop:
    p = opt_params(opt_state)
    params = [p[0],p[1],p[2], all_cameras,beta_2,beta_3,beta_4,beta_5]
    val,g = grad_render3(params, all_sils)
    opt_state = opt_update(i, g[:3], opt_state)

    means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5 = params
    depth,_,alpha = fm_render.render(means,prec,weights_log,cameras_list[cam_idx],beta_2,beta_3,beta_4,beta_5)

    depth = depth.at[alpha < 0.5].set(0.0)
    optimization_depths.append(depth)

   
    val = float(val)
    losses.append(val)
    loop.set_description("total_loss = %.3f" % val)


min_depth=1.0
max_depth=10.0
ground_truth_depth = all_depths[cam_idx]
ground_truth_alpha = alpha_list[cam_idx]
ground_truth_depth_viz = ground_truth_depth.at[ground_truth_alpha < 0.5].set(0.0)

ground_truth_img = get_depth_image(ground_truth_depth_viz.reshape(image_size), min=min_depth, max=max_depth)


optimization_images = [
    multi_panel(
        [
            ground_truth_img.resize(image_size_visualization),
            get_depth_image(depth.reshape(image_size),min=min_depth, max=max_depth).resize(image_size_visualization)
        ],
        ["Ground Truth", "Current Shape Iteration {:03d}".format(i)],
        10,
        50,
        20
    )
    for (i,depth) in enumerate(optimization_depths)
]


p = opt_params(opt_state)


optimization_images[0].save(
    fp="out.gif",
    format="GIF",
    append_images=optimization_images,
    save_all=True,
    duration=100,
    loop=0,
)



from IPython import embed; embed()
