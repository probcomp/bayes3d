from jax3dp3.fuzzy import fuzzy_renderer as fm_render
import jax.numpy as jnp
import jax
import numpy as np
from jax3dp3.viz.img import save_depth_image,save_rgb_image, get_depth_image, multi_panel
from tqdm import tqdm

image_size = (64,64)
height, width = image_size
cx = (width-1)/2
cy = (height-1)/2

vfov_degrees = 60

f = (height/np.tan((np.pi/180)*float(vfov_degrees)/2))*0.5
K = np.array([[f, 0, cx],[0,f,cy],[0,0,1]])
pixel_list = (np.array(np.meshgrid(width-np.arange(width)-1,height-np.arange(height)-1,[0]))[:,:,:,0]).reshape((3,-1)).T

camera_rays = (pixel_list - K[:,2])/np.diag(K)
camera_rays[:,-1] = 1

R = np.eye(3)
T = np.zeros(3)

translation = np.array(-R @ T)

camera_rays = camera_rays @ np.array(R ).T 
trans = np.tile(translation[None],(camera_rays.shape[0],1))

camera_starts_rays = np.stack([camera_rays,trans],1)

hyperparams = fm_render.hyperparams
NUM_MIXTURE = 40
beta_2 = jnp.float32(np.exp(hyperparams[0]))
beta_3 = jnp.float32(np.exp(hyperparams[1]))
beta_4 = jnp.float32(np.exp(hyperparams[2]))
beta_5 = -jnp.float32(np.exp(hyperparams[3]))


center = jnp.array([0.0, 0.0, 5.0])
shape_scale = 2.0
rand_mean = center+np.random.multivariate_normal(mean=[0,0,0],cov=1e-1*np.identity(3)*shape_scale,size=NUM_MIXTURE)
rand_weight_log = jnp.log(np.ones(NUM_MIXTURE)/NUM_MIXTURE)
rand_sphere_size = 14
rand_prec = jnp.array([np.identity(3)*rand_sphere_size/shape_scale for _ in range(NUM_MIXTURE)])

init_t,stds,est_alpha = fm_render.render(rand_mean, rand_prec, rand_weight_log, camera_starts_rays, beta_2, beta_3, beta_4, beta_5)

print('est_alpha.max():');print(est_alpha.max())
print('est_alpha.min():');print(est_alpha.min())

ground_truth_alpha = est_alpha

max_depth = 1.0
ground_truth_img = get_depth_image(ground_truth_alpha.reshape(image_size), max_depth)



rand_mean = center+np.random.multivariate_normal(mean=[0,0,0],cov=1e-1*np.identity(3)*shape_scale,size=NUM_MIXTURE)
rand_weight_log = jnp.log(np.ones(NUM_MIXTURE)/NUM_MIXTURE)
rand_prec = jnp.array([np.identity(3)*rand_sphere_size/shape_scale for _ in range(NUM_MIXTURE)])

init_t,stds,est_alpha = fm_render.render(rand_mean, rand_prec, rand_weight_log, camera_starts_rays, beta_2, beta_3, beta_4, beta_5)

initial_img = get_depth_image(est_alpha.reshape(image_size),max_depth)


def objective(params,true_alpha):
    CLIP_ALPHA = 1e-6
    means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5 = params
    render_res = fm_render.render(means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5)

    est_alpha = render_res[2]
    est_alpha = jnp.clip(est_alpha,CLIP_ALPHA,1-CLIP_ALPHA)
    mask_loss = - ((true_alpha * jnp.log(est_alpha)) + (1-true_alpha)*jnp.log(1-est_alpha))
    return mask_loss.mean()
grad_render3 = jax.jit(jax.value_and_grad(objective))


from jax.example_libraries import optimizers

# Number of optimization steps
Niter = 200
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

# for i in loop:
p = opt_params(opt_state)
val,g = grad_render3([p[0],p[1],p[2], camera_starts_rays,beta_2,beta_3,beta_4,beta_5], ground_truth_alpha)

losses = []
accum_grad = None
grad_counter = 0

optimization_images = []
scaling_factor = 4
image_size_visualization = (image_size[0]*scaling_factor, image_size[1]*scaling_factor)
for i in loop:
    p = opt_params(opt_state)
    params = [p[0],p[1],p[2], camera_starts_rays,beta_2,beta_3,beta_4,beta_5]
    val,g = grad_render3(params, ground_truth_alpha)
    opt_state = opt_update(i, g[:3], opt_state)

    means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5 = params
    _,_,est_alpha = fm_render.render(means,prec,weights_log,camera_rays,beta_2,beta_3,beta_4,beta_5)
    img = get_depth_image(est_alpha.reshape(image_size),max_depth)
    optimization_images.append(
        multi_panel(
            [ground_truth_img.resize(image_size_visualization), img.resize(image_size_visualization)],
            ["Ground Truth", "Current Shape Iteration {:03d}".format(i)],
            10,
            50,
            20
        )
    )
   
    val = float(val)
    losses.append(val)
    loop.set_description("total_loss = %.3f" % val)

optimization_images[0].save(
    fp="out.gif",
    format="GIF",
    append_images=optimization_images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()