import machine_common_sense as mcs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jax
import time
from jax3dp3.viz import save_depth_image,save_rgb_image
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
import cv2
import jax.numpy as jnp
from jax3dp3.bbox import axis_aligned_bounding_box
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.rendering import render_planes_multiobject
from jax3dp3.enumerations import make_translation_grid_enumeration
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.viz import multi_panel
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.distributions import gaussian_vmf, gaussian_vmf_cov
from jax.scipy.special import logsumexp
from jax3dp3.rendering import render_spheres, render_cloud_at_pose,render_planes_multiobject
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_imgs_original = data["depth_images"]
seg_images_original = data["seg_images"]

scaling_factor = 0.25

fx = data["fx"] * scaling_factor
fy = data["fy"] * scaling_factor

cx = data["cx"] * scaling_factor
cy = data["cy"] * scaling_factor

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])
original_height = data["height"]
original_width = data["width"]
h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)


coord_images = []
seg_images = []
for (d,s) in zip(depth_imgs_original, seg_images_original):
    coord_images.append(
        depth_to_coords_in_camera(cv2.resize(d.copy(), (w,h),interpolation=0), K.copy())[0]
    )
    seg_images.append(
        cv2.resize(s.copy(), (w,h),interpolation=0)
    )

coord_images = np.stack(coord_images)
seg_images = np.stack(seg_images)
coord_images = np.concatenate([coord_images, np.ones(coord_images.shape[:3])[:,:,:,None] ], axis=-1)
fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])
coord_images = jnp.array(coord_images)

start_t = 92
end_t = start_t + 20
mask = (coord_images[:, :,:,2] < 10.0) * (coord_images[:,:,:,1] < 1.46) 
save_depth_image(coord_images[start_t,:,:,2] * mask[start_t], 30.0, "out.png")

save_rgb_image(seg_images[start_t, :,:, :], 255.0, "rgb.png")
save_rgb_image(seg_images[start_t, :,:, :] * mask[start_t,:,:,None], 255.0, "rgb2.png")
vals = jnp.unique(seg_images[start_t, :,:, 2] * mask[start_t])
vals = vals[vals > 0]

shape_planes = []
shape_dims = []
initial_poses = []
for val in vals:
    object_mask = (seg_images[start_t, :,:,2] == val) * mask[start_t]
    dims, pose = axis_aligned_bounding_box(coord_images[start_t,object_mask,:3])
    plane, dims = get_rectangular_prism_shape(dims + 0.0)
    initial_poses.append(pose)
    shape_planes.append(plane)
    shape_dims.append(dims)
initial_poses = jnp.array(initial_poses)
shape_dims = jnp.array(shape_dims)
shape_planes = jnp.array(shape_planes)

img = render_planes_multiobject(initial_poses, shape_planes, shape_dims, h,w, fx_fy, cx_cy)
save_depth_image(img[:,:,2], 30.0, "rendering.png")

render_planes_multiobject_jit = jax.jit(lambda p: render_planes_multiobject(p, shape_planes, shape_dims, h,w, fx_fy, cx_cy))

r = 0.1
outlier_prob = 0.1
def likelihood(x, obs):
    rendered_image = render_planes_multiobject(x, shape_planes, shape_dims, h,w, fx_fy, cx_cy)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight
likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)

depth_data = (coord_images * mask[:,:,:,None])

cm = plt.get_cmap("turbo")
max_depth = 30.0
middle_width = 20
top_border = 100


object_index = 2

def run_inference(initial_particles, initial_particles_velocities, ground_truth_images):
    variances = jnp.stack([
        jnp.diag(jnp.array([0.06, 0.0001, 0.0001])),
        jnp.diag(jnp.array([0.06, 0.0001, 0.0001])),
        jnp.diag(jnp.array([0.06, 0.0001, 0.0001])),
    ]
    )
        # jnp.array(0.05, 0.3, 0.5])
    concentrations = jnp.array([2000.0, 2000.0, 2000.0])
    mixture_logits = jnp.log(jnp.ones(concentrations.shape) / concentrations.shape[0])

    def particle_filtering_step(data, gt_image):
        particles, particle_velocities, weights, keys = data
        proposal_type = jax.random.categorical(keys[0], mixture_logits, shape=(particles.shape[0],))
        drift_poses = jax.vmap(gaussian_vmf_cov, in_axes=(0, 0, 0))(keys, variances[proposal_type], concentrations[proposal_type])
        i = object_index
        new_particles = particles.copy()
        new_particles = new_particles.at[:, i, :3, -1].set(new_particles[:, i, :3, -1] + particle_velocities)
        new_particles = new_particles.at[:, i].set(jnp.einsum("...ij,...jk->...ik", new_particles[:,i], drift_poses))

        new_particles_velocities = new_particles[:, i, :3, -1] - particles[:, i, :3, -1]

        weights = weights + likelihood_parallel(new_particles, gt_image)
        parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
        new_particles = new_particles[parent_idxs]
        new_particles_velocities = new_particles_velocities[parent_idxs]
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))
        keys = jax.random.split(keys[0], weights.shape[0])
        return (new_particles, new_particles_velocities, weights, keys), particles

    initial_weights = jnp.full(initial_particles.shape[0], 0.0)
    initial_key = jax.random.PRNGKey(3)
    initial_keys = jax.random.split(initial_key, initial_particles.shape[0])
    return jax.lax.scan(particle_filtering_step, (initial_particles, initial_particles_velocities, initial_weights, initial_keys), ground_truth_images)


run_inference_jit = jax.jit(run_inference)
num_particles = 1000
particle_positions = []
particle_velocities = []
for _ in range(num_particles):
    particle_positions.append(initial_poses)
    particle_velocities.append(np.zeros(3))

particle_positions = jnp.stack(particle_positions)
particle_velocities = jnp.stack(particle_velocities)

run_inference_jit = jax.jit(run_inference)
end_t = start_t + 50
_,x = run_inference_jit(particle_positions,  particle_velocities, depth_data[start_t:end_t])

start = time.time()
_,x = run_inference_jit(particle_positions,  particle_velocities, depth_data[start_t:end_t])
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", (depth_data[start_t:end_t].shape[0]) / (end-start))



cm = plt.get_cmap("turbo")
images = []
for i in range(start_t, end_t):
    rgb = rgb_images[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGB"
    )
    depth_img = Image.fromarray(
        np.rint(
            cm(np.array(depth_data[i, :, :, 2]) / max_depth) * 255.0
        ).astype(np.int8),
        mode="RGBA",
    ).resize((original_width,original_height))
    
    poses = x[i-start_t,0,:,:,:]
    rendered_image = render_planes_multiobject_jit(poses)
    rendered_depth_img = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    ).resize((original_width,original_height))

    i1 = rendered_depth_img.copy()
    i2 = rgb_img.copy()
    i1.putalpha(128)
    i2.putalpha(128)
    overlay_img = Image.alpha_composite(i1, i2)

    translations = x[i-start_t, :, object_index, :3, -1]
    img = render_cloud_at_pose(translations, jnp.eye(4), h, w, fx_fy, cx_cy, 0)
    projected_particles_img = get_depth_image(img[:,:,2], 40.0).resize((original_width,original_height))

    panel_images = [rgb_img, depth_img, rendered_depth_img, overlay_img, projected_particles_img]
    labels = ["RGB Image", "Depth Image", "Inferred Depth", "Overlay", "Particles"]
    dst = multi_panel(panel_images, labels, middle_width, top_border, 40)
    images.append(dst)


images[0].save(
    fp="out.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)





from IPython import embed; embed()
