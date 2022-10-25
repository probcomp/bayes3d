import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.likelihood import neural_descriptor_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix
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
key = jax.random.PRNGKey(3)

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


key = jax.random.PRNGKey(3)
def scorer(key, pose, gt_image):
    rendered_image = render_from_pose(pose)
    weight = neural_descriptor_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes = (0, 0, None))


f_jit = jax.jit(jax.vmap(lambda t:     jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )))
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.2, 0.2, 0.2, 5, 5, 5))
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

categorical_vmap = jax.vmap(jax.random.categorical, in_axes=(None, 0))
logsumexp_vmap = jax.vmap(logsumexp)


def run_inference(initial_pose, ground_truth_images):
    def _inner(x, gt_image):
        for _ in range(10):
            proposals = jnp.einsum("ij,ajk->aik", x, pose_deltas)
            weights_new = scorer_parallel(sub_keys_translation, proposals, gt_image)
            x = proposals[jnp.argmax(weights_new)]

            proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
            weights_new = scorer_parallel(sub_keys_orientation, proposals, gt_image)
            x = proposals[jnp.argmax(weights_new)]

        return x, x

    return jax.lax.scan(_inner, initial_pose, ground_truth_images)


run_inference_jit = jax.jit(run_inference)
_,x = run_inference_jit(first_pose, ground_truth_images)

from IPython import embed; embed()


start = time.time()
_,x = run_inference_jit(first_pose, ground_truth_images)
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


    pose = x[i,:,:]
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