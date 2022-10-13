import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes, render_sphere
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.likelihood import neural_descriptor_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.viz.gif import make_gif
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2


data = np.load("ball.npz")
depth_imgs = np.array(data["depth_images"]).copy() * 0.001
rgb_images = np.array(data["rgb_images"]).copy() 

original_fx, original_fy =  385.798, 385.798
original_cx, original_cy = 321.49, 244.092
original_height = 480
original_width = 640

scaling_factor = 0.25

fx_fy = jnp.array([original_fx * scaling_factor, original_fy * scaling_factor])
cx_cy = jnp.array([original_cx * scaling_factor, original_cy * scaling_factor])

K = np.array([
    [fx_fy[0], 0.0, cx_cy[0]],
    [0.0, fx_fy[1], cx_cy[1]],
    [0.0, 0.0, 1.0],
])

h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))

coord_images = [
    depth_to_coords_in_camera(
        cv2.resize(
            cv2.bilateralFilter(d.copy().astype(np.float32), 4, 1.0, 1.0),
            (w,h),interpolation=1
        ),
        K.copy()
    )[0]
    for d in depth_imgs
]
gt_images = np.stack(coord_images)
print('gt_images.shape ',gt_images.shape)
gt_images = gt_images[160:190,:,:,:]
rgb_images = rgb_images[160:190,:,:,:]

make_gif(gt_images, 3.0, "imgs/rgb_real.gif")

gt_images[gt_images[:,:,:,2] > 1.3] = 0.0   
gt_images[gt_images[:,:,:,2] < 0.2] = 0.0
gt_images[gt_images[:,:,:,1] > 0.1,:] = 0.0
gt_images = np.concatenate([gt_images, np.ones(gt_images.shape[:3])[:,:,:,None] ], axis=-1)
print('gt_images.shape ',gt_images.shape)

make_gif(gt_images, 3.0, "imgs/rgb_real_filtered.gif")


r = 0.01
outlier_prob = 0.2

initial_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.6],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
gt_images = jnp.array(gt_images)

radius = 0.1
key = jax.random.PRNGKey(3)
def scorer(key, pose, gt_image):
    rendered_image = render_sphere(pose, radius, h, w, fx_fy, cx_cy)
    weight = neural_descriptor_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight


score = scorer(key, initial_pose, gt_images[0,:,:,:])
print("score", score)

scorer_parallel = jax.vmap(scorer, in_axes = (0, 0, None))

f_jit = jax.jit(jax.vmap(lambda t:     jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )))
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.1, 0.1, 0.1, 8, 8, 8))
print("grid ", pose_deltas.shape)
key, *sub_keys = jax.random.split(key, pose_deltas.shape[0] + 1)
sub_keys_translation = jnp.array(sub_keys)

key, *sub_keys = jax.random.split(key, 300)
sub_keys = jnp.array(sub_keys)
def f(key):
    v = VonMisesFisher(
        jnp.array([1.0, 0.0, 0.0, 0.0]), 1000.0
    ).sample(seed=key)
    r =  quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([r, jnp.zeros((3, 1)) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )
f_jit = jax.jit(jax.vmap(f))
rotation_deltas = f_jit(sub_keys)
print("grid ", rotation_deltas.shape)
key, *sub_keys = jax.random.split(key, rotation_deltas.shape[0] + 1)
sub_keys_orientation = jnp.array(sub_keys)

render_planes_jit = jax.jit(lambda p:  render_sphere(p, radius, h, w, fx_fy, cx_cy))
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_sphere(p, radius, h, w, fx_fy, cx_cy)))

def _inner(x, gt_image):
    for _ in range(1):
        proposals = jnp.einsum("ij,ajk->aik", x, pose_deltas)
        weights_new = scorer_parallel(sub_keys_translation, proposals, gt_image)
        x = proposals[jnp.argmax(weights_new)]

        proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
        weights_new = scorer_parallel(sub_keys_orientation, proposals, gt_image)
        x = proposals[jnp.argmax(weights_new)]

    return x, x


def inference(init_pos, gt_images):
    return jax.lax.scan(_inner, init_pos, gt_images)



inference_jit = jax.jit(inference)

a = inference_jit(initial_pose, gt_images)

start = time.time()
_, inferred_poses = inference_jit(initial_pose, gt_images);
end = time.time()
print ("Time elapsed:", end - start)


cm = plt.get_cmap('turbo')

max_depth = 3.0
images = []
middle_width = 50
for i in range(gt_images.shape[0]):
    dst = Image.new(
        "RGBA", (3 * w + 2*middle_width, h)
    )
    rgb = rgb_images[i]
    rgb_img = Image.fromarray(
        rgb[:,:,::-1].astype(np.int8), mode="RGB"
    ).resize((w,h)).convert("RGBA")
    dst.paste(
        rgb_img,
        (0, 0),
    )

    obsedved_image_pil = Image.fromarray(
        (cm(np.array(gt_images[i,:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    )

    dst.paste(
        obsedved_image_pil,
        (w+middle_width, 0),
    )

    pose = inferred_poses[i]
    rendered_image = render_planes_jit(pose)
    rendered_image_pil = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    )

    dst.paste(
        rendered_image_pil,
        (2*w + 2*middle_width, 0),
    )
    images.append(dst)



images[0].save(
    fp="imgs/out.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()