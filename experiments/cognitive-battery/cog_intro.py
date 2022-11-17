import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from jax3dp3.viz.img import save_depth_image
from jax3dp3.utils import depth_to_coords_in_camera
from jax3dp3.transforms_3d import transform_from_pos
from jax3dp3.shape import (
    get_rectangular_prism_shape,
)
from jax3dp3.likelihood import threedp3_likelihood
import jax.numpy as jnp

import jax
from scipy.spatial.transform import Rotation as R
from jax3dp3.rendering import render_planes, render_planes_multiobject
from jax3dp3.enumerations import make_translation_grid_enumeration
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame

# Initialize metadata

num_frames = 103
data_path = "data/videos"

width =  300
height =  300
fx =  150
fy =  150
cx =  150
cy =  150

fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx, cy])

# Load and pre-process rgb and depth images

rgb_images, depth_images, seg_maps = [], [], [] 
rgb_images_pil = []
for i in range(num_frames):
    rgb_path = os.path.join(data_path, f"frames/frame_{i}.jpeg")
    rgb_img = Image.open(rgb_path)
    rgb_images_pil.append(rgb_img)
    rgb_images.append(np.array(rgb_img))

    depth_path = os.path.join(data_path, f"depths/frame_{i}.npy")
    depth_npy = np.load(depth_path)
    depth_images.append(depth_npy)

    seg_map = np.load(os.path.join(data_path, f"segmented/frame_{i}.npy"))
    seg_maps.append(seg_map)
    
    
    
# Get masked objects based on the depth images

frame_idx = 50
k = 5 if 5 <= frame_idx < 19 else 4 # 4 objects in frames [5:19]

K = jnp.array([
    [fx_fy[0], 0.0, cx_cy[0]],
    [0.0, fx_fy[1], cx_cy[1]],
    [0.0, 0.0, 1.0],
])
coord_image,_ = depth_to_coords_in_camera(depth_images[frame_idx], K)
segmentation_image = seg_maps[frame_idx]
print(coord_image.shape)
# -.5 < x < 1
# -.5 < y < .5
# 1.2 < z < 4
mask = np.invert(
    (coord_image[:,:,0] < 1.0) *
    (coord_image[:,:,0] > -0.5) *
    (coord_image[:,:,1] < 0.28) *
    (coord_image[:,:,1] > -0.5) *
    (coord_image[:,:,2] < 4.0) *
    (coord_image[:,:,2] > 1.2) 
)
coord_image[mask,:] = 0.0
save_depth_image(coord_image[:,:,2], 30.0, "coord_image.png")


# Do K-Means clustering to segment objects from masked depth (i.e. entity extraction)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .02)

coord_image_flat = coord_image.reshape(-1, 3).astype(dtype=np.float32)
_, labels, centers = cv2.kmeans(coord_image_flat, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)



_a = labels.reshape(300, 300)
_img = np.stack((_a, _a, _a), axis=-1)
# save_depth_image(_a, 5.0, "seg_map.png")


shape_planes, shape_dims, init_poses = [], [], []

for obj_id in range(k):
    obj_mask = (_a == obj_id)

    masked_coord_image = coord_image * obj_mask[:,:,None]
    masked_segmentation_image = segmentation_image * obj_mask[:,:,None]
    # save_depth_image(masked_coord_image[:,:,2], 5.0, "seg_map_single_entity.png")
    
    object_points = masked_coord_image[obj_mask]
    maxs = np.max(object_points,axis=0)
    mins = np.min(object_points,axis=0)
    dims = (maxs - mins)
    center_of_box = (maxs+mins)/2
    
    init_pose = transform_from_pos(center_of_box)
    init_poses.append(init_pose)

    shape, dim = get_rectangular_prism_shape(dims)
    shape_planes.append(shape)
    shape_dims.append(dim)
    # break
    
shape_planes = jnp.stack(shape_planes)
shape_dims = jnp.stack(shape_dims)
init_poses = jnp.stack(init_poses)

reconstruction_image = render_planes_multiobject(
    init_poses,
    shape_planes,
    shape_dims,
    height,
    width,
    fx_fy,
    cx_cy
)
save_depth_image(reconstruction_image[:,:,2], 5.0, "reconstruction_multi.png")
    

# Get masked coord_images and seg_images

coord_images = []
seg_images = []

for frame_idx in range(20, num_frames):
    # frame_idx = 20
    k = 5 if 5 <= frame_idx < 19 else 4 # 4 objects in frames [5:19]

    K = jnp.array([
        [fx_fy[0], 0.0, cx_cy[0]],
        [0.0, fx_fy[1], cx_cy[1]],
        [0.0, 0.0, 1.0],
    ])
    coord_image,_ = depth_to_coords_in_camera(depth_images[frame_idx], K)
    segmentation_image = seg_maps[frame_idx]
    mask = np.invert(
        (coord_image[:,:,0] < 1.0) *
        (coord_image[:,:,0] > -0.5) *
        (coord_image[:,:,1] < 0.28) *
        (coord_image[:,:,1] > -0.5) *
        (coord_image[:,:,2] < 4.0) *
        (coord_image[:,:,2] > 1.2) 
    )
    coord_image[mask,:] = 0.0
    segmentation_image[mask,:] = 0.0
    coord_images.append(coord_image)
    seg_images.append(segmentation_image)

coord_images = np.stack(coord_images)
seg_images = np.stack(seg_images)

render_planes_multiobject_jit = jax.jit(lambda p: render_planes_multiobject(p, shape_planes, shape_dims, height, width, fx_fy, cx_cy))

r = 0.1
outlier_prob = 0.1
def likelihood(x, obs):
    rendered_image = render_planes_multiobject(x, shape_planes, shape_dims, height, width, fx_fy, cx_cy)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight

likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)


enumerations = make_translation_grid_enumeration(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 51, 21, 11)

cm = plt.get_cmap("turbo")
max_depth = 30.0
middle_width = 20
top_border = 100

start_t = 20
x = init_poses
t = start_t + 1
gt_image = coord_images[t]
for i in range(x.shape[0]):
    enumerations_full = jnp.tile(jnp.eye(4)[None, :,:],(enumerations.shape[0], x.shape[0],1,1))
    enumerations_full = enumerations_full.at[:,i,:,:].set(enumerations)
    proposals = jnp.einsum("bij,abjk->abik", x, enumerations_full)

    proposals_batched = jnp.stack(jnp.split(proposals, 21))
    x = enumerative_inference_single_frame(likelihood_parallel, gt_image, proposals_batched)[0]
    
    # break


rgb = rgb_images[t]
rgb_img = Image.fromarray(
    rgb.astype(np.int8), mode="RGB"
)
depth_img = Image.fromarray(
    np.rint(
        cm(np.array(coord_images[t, :, :, 2]) / max_depth) * 255.0
    ).astype(np.int8),
    mode="RGBA",
).resize((height, width))