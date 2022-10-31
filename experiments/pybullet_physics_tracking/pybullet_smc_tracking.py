import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax3dp3.distributions import gaussian_vmf
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from jax.scipy.special import logsumexp
from jax3dp3.viz.gif import make_gif


data = np.load("data.npz")
depth_imgs = np.array(data["depth_imgs"]).copy()
rgb_imgs = np.array(data["rgb_imgs"]).copy()

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

coord_images = [depth_to_coords_in_camera(cv2.resize(d.copy(), (w,h),interpolation=1), K.copy())[0] for d in depth_imgs]
ground_truth_images = np.stack(coord_images)
ground_truth_images[ground_truth_images[:,:,:,2] > 40.0] = 0.0
ground_truth_images[ground_truth_images[:,:,:,1] > 0.85,:] = 0.0
ground_truth_images = np.concatenate([ground_truth_images, np.ones(ground_truth_images.shape[:3])[:,:,:,None] ], axis=-1)
fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx,cy])
ground_truth_images = jnp.array(ground_truth_images)


r = 0.1
outlier_prob = 0.1
first_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, -5.00],
        [0.0, 1.0, 0.0, -4.00],
        [0.0, 0.0, 1.0, 20.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

shape = get_cube_shape(2.0)

render_from_pose = lambda pose: render_planes(pose,shape,h,w,fx_fy,cx_cy)
render_from_pose_jit = jax.jit(render_from_pose)
render_planes_parallel_jit = jax.jit(jax.vmap(lambda x: render_from_pose(x)))


def likelihood(x, obs):
    rendered_image = render_from_pose(x)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight
likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)

categorical_vmap = jax.vmap(jax.random.categorical, in_axes=(None, 0))
logsumexp_vmap = jax.vmap(logsumexp)

DRIFT_VAR = 0.2

def run_inference(initial_particles, ground_truth_images):
    def particle_filtering_step(data, gt_image):
        particles, weights, keys = data
        drift_poses = jax.vmap(gaussian_vmf, in_axes=(0, None, None))(keys, DRIFT_VAR, 100.0)
        particles = jnp.einsum("...ij,...jk->...ik", particles, drift_poses)
        weights = weights + likelihood_parallel(particles, gt_image)
        parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
        particles = particles[parent_idxs]
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))
        keys = jax.random.split(keys[0], weights.shape[0])
        return (particles, weights, keys), particles

    initial_weights = jnp.full(initial_particles.shape[0], 0.0)
    initial_key = jax.random.PRNGKey(3)
    initial_keys = jax.random.split(initial_key, initial_particles.shape[0])
    return jax.lax.scan(particle_filtering_step, (initial_particles, initial_weights, initial_keys), ground_truth_images)


run_inference_jit = jax.jit(run_inference)
num_particles = 1000
particles = []
for _ in range(num_particles):
    particles.append(first_pose)
particles = jnp.stack(particles)
_,x = run_inference_jit(particles, ground_truth_images)


start = time.time()
_,x = run_inference_jit(particles, ground_truth_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", ground_truth_images.shape[0] / (end - start))




max_depth = 30.0
middle_width = 20
cm = plt.get_cmap("turbo")
images = []
for i in range(ground_truth_images.shape[0]):
    dst = Image.new(
        "RGBA", (3 * original_width + 2*middle_width, original_height)
    )

    rgb = rgb_imgs[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )
    dst.paste(
        rgb_img,
        (0,0)
    )

    dst.paste(
        Image.new(
            "RGBA", (middle_width, original_height), (255, 255, 255, 255)
        ),
        (original_width, 0),
    )

    dst.paste(
        Image.fromarray(
            np.rint(
                cm(np.array(ground_truth_images[i, :, :, 2]) / max_depth) * 255.0
            ).astype(np.int8),
            mode="RGBA",
        ).resize((original_width,original_height)),
        (original_width + middle_width, 0),
    )

    dst.paste(
        Image.new(
            "RGBA", (middle_width, original_height), (255, 255, 255, 255)
        ),
        (2* original_width + middle_width, 0),
    )


    pose = x[i,-1,:,:]
    rendered_image = render_from_pose_jit(pose)
    overlay_image_1 = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    ).resize((original_width,original_height))
    overlay_image_1.putalpha(128)
    rgb_img_copy = rgb_img.copy()
    rgb_img_copy.putalpha(128)

    dst.paste(
        Image.alpha_composite(overlay_image_1, rgb_img_copy),
        (2*original_width + 2*middle_width, 0),
    )
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