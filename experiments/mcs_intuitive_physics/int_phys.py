import machine_common_sense as mcs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jax
from jax3dp3.viz.img import save_depth_image,save_rgb_image
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
from jax3dp3.viz.img import multi_panel
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame

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

enumerations = make_translation_grid_enumeration(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 51, 21, 11)

depth_data = (coord_images * mask[:,:,:,None])

cm = plt.get_cmap("turbo")
max_depth = 30.0
middle_width = 20
top_border = 100


x = initial_poses
t = start_t + 1
gt_image = depth_data[t]
for i in range(x.shape[0]):
    enumerations_full = jnp.tile(jnp.eye(4)[None, :,:],(enumerations.shape[0], x.shape[0],1,1))
    enumerations_full = enumerations_full.at[:,i,:,:].set(enumerations)
    proposals = jnp.einsum("bij,abjk->abik", x, enumerations_full)

    proposals_batched = jnp.stack(jnp.split(proposals, 21))
    x = enumerative_inference_single_frame(likelihood_parallel, gt_image, proposals_batched)[0]

rgb = rgb_images[t]
rgb_img = Image.fromarray(
    rgb.astype(np.int8), mode="RGB"
)
depth_img = Image.fromarray(
    np.rint(
        cm(np.array(depth_data[t, :, :, 2]) / max_depth) * 255.0
    ).astype(np.int8),
    mode="RGBA",
).resize((original_width,original_height))

poses = initial_poses
rendered_image = render_planes_multiobject_jit(poses)
rendered_depth_img_initial = Image.fromarray(
    (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
).resize((original_width,original_height))

poses = x
rendered_image = render_planes_multiobject_jit(poses)
rendered_depth_img = Image.fromarray(
    (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
).resize((original_width,original_height))

i1 = rendered_depth_img.copy()
i2 = rgb_img.copy()
i1.putalpha(128)
i2.putalpha(128)
overlay_img = Image.alpha_composite(i1, i2)

i1 = rendered_depth_img_initial.copy()
i2 = rgb_img.copy()
i1.putalpha(128)
i2.putalpha(128)
overlay_img_original = Image.alpha_composite(i1, i2)

panel_images = [rgb_img, depth_img, rendered_depth_img_initial, rendered_depth_img, overlay_img_original, overlay_img]
labels = ["RGB Image", "Depth Image", "Initial", "Inferred Depth", "Overlay Original", "Overlay"]
dst = multi_panel(panel_images, labels, middle_width, top_border, 40)
dst.save("initial.png")






def run_inference(initial_poses, ground_truth_images):
    def _inner(x, gt_image):
        for i in range(x.shape[0]):
            enumerations_full = jnp.tile(jnp.eye(4)[None, :,:],(enumerations.shape[0], x.shape[0],1,1))
            enumerations_full = enumerations_full.at[:,i,:,:].set(enumerations)
            proposals = jnp.einsum("bij,abjk->abik", x, enumerations_full)

            proposals_batched = jnp.stack(jnp.split(proposals, 21))
            x = enumerative_inference_single_frame(likelihood_parallel, gt_image, proposals_batched)[0]

        return x, x

    return jax.lax.scan(_inner, initial_poses, ground_truth_images)

run_inference_jit = jax.jit(run_inference)
end_t = start_t + 20
_, tracked_poses = run_inference_jit(initial_poses, depth_data[start_t:end_t])


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
    
    poses = tracked_poses[i-start_t,:,:,:]
    rendered_image = render_planes_multiobject_jit(poses)
    rendered_depth_img = Image.fromarray(
        (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    ).resize((original_width,original_height))

    i1 = rendered_depth_img.copy()
    i2 = rgb_img.copy()
    i1.putalpha(128)
    i2.putalpha(128)
    overlay_img = Image.alpha_composite(i1, i2)

    panel_images = [rgb_img, depth_img, rendered_depth_img, overlay_img]
    labels = ["RGB Image", "Depth Image", "Inferred Depth", "Overlay"]
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
