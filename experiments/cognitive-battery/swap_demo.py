import os
import numpy as np
import matplotlib.pyplot as plt
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel
from jax3dp3.utils import depth_to_coords_in_camera
from jax3dp3.transforms_3d import transform_from_pos
from jax3dp3.shape import (
    get_rectangular_prism_shape,
)
from jax3dp3.likelihood import threedp3_likelihood
import jax.numpy as jnp
import jax
from scipy.spatial.transform import Rotation as R
from jax3dp3.rendering import render_planes_multiobject
from jax3dp3.enumerations import make_translation_grid_enumeration
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.viz.gif import make_gif_from_pil_images
from PIL import Image

from tqdm import tqdm

# Initialize metadata
def get_camera_intrinsics(width, height, fov):
    cx, cy = width / 2.0, height / 2.0
    aspect_ratio = width / height
    fov_y = np.deg2rad(fov)
    fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
    fx = cx / np.tan(fov_x / 2.0)
    fy = cy / np.tan(fov_y / 2.0)
    
    return fx, fy, cx, cy

data_path = "data/swap_data/videos"
num_frames = len(os.listdir(os.path.join(data_path, "frames")))

width =  300
height =  300

fov = 99

if fov:
    fx, fy, cx, cy = get_camera_intrinsics(width, height, fov)
else:
    fx =  150
    fy =  150
    cx =  150
    cy =  150

fx_fy = jnp.array([fx, fy])
cx_cy = jnp.array([cx, cy])

K = jnp.array([
    [fx_fy[0], 0.0, cx_cy[0]],
    [0.0, fx_fy[1], cx_cy[1]],
    [0.0, 0.0, 1.0],
])

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
    
coord_images = []
seg_images = []

for frame_idx in range(num_frames):
    coord_image,_ = depth_to_coords_in_camera(depth_images[frame_idx], K)
    segmentation_image = seg_maps[frame_idx]
    mask = np.invert(
        (coord_image[:,:,0] < 1.0) *
        (coord_image[:,:,0] > -1) *
        (coord_image[:,:,1] < 0.465) *
        (coord_image[:,:,1] > -.8) *
        (coord_image[:,:,2] < 2) *
        (coord_image[:,:,2] > 1.15) 
    )
    coord_image[mask,:] = 0.0
    segmentation_image[mask,:] = 0.0
    coord_images.append(coord_image)
    seg_images.append(segmentation_image)

coord_images = np.stack(coord_images)
seg_images = np.stack(seg_images)

start_t = 0
shape_planes, shape_dims, init_poses = [], [], []
seg_img = seg_images[start_t][:,:,2]

imgs = []
for obj_id in jnp.unique(seg_img):
    imgs.append(get_depth_image(seg_img == obj_id))
multi_panel(imgs, [], 10, 10, 13).save("masks.png")

for obj_id in jnp.unique(seg_img):
    if obj_id == 0:
        continue
    obj_mask = (seg_img == obj_id)
    
    object_points = coord_images[start_t][obj_mask]
    maxs = np.max(object_points,axis=0)
    mins = np.min(object_points,axis=0)
    dims = (maxs - mins)
    center_of_box = (maxs+mins)/2
    
    init_pose = transform_from_pos(center_of_box)
    init_poses.append(init_pose)

    shape, dim = get_rectangular_prism_shape(dims)
    shape_planes.append(shape)
    shape_dims.append(dim)
    
shape_planes = jnp.stack(shape_planes)
shape_dims = jnp.stack(shape_dims)
init_poses = jnp.stack(init_poses)

reconstruction_image = render_planes_multiobject(
    init_poses,
    shape_planes,
    shape_dims,
    height, width, fx,fy, cx, cy
)

def render_planes_multiobject_lambda(poses):
    return (
        render_planes_multiobject(poses, shape_planes, shape_dims, height, width, fx,fy, cx,cy)
    )
render_planes_multiobject_jit = jax.jit(render_planes_multiobject_lambda)

reconstruction_image = render_planes_multiobject_jit(init_poses)
# save_depth_image(reconstruction_image[:,:,2], "reconstruction.png", max=5.0)

r = 0.01
outlier_prob = 0.01
def likelihood(x, params):
    obs= params[0]
    rendered_image = render_planes_multiobject(x, shape_planes, shape_dims, height, width, fx,fy, cx,cy)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight

likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)

n = num_proposals = 7
d = delta = 0.1

enumerations = make_translation_grid_enumeration(-d, -d, -d, d, d, d, n, n, n)

cm = plt.get_cmap("turbo")
max_depth = 30.0
middle_width = 20
top_border = 100


pose_estimates = init_poses.copy()

inferred_poses = []
batched_scorer_parallel_jit = jax.jit(lambda poses, image: batched_scorer_parallel(likelihood_parallel, n, poses, (image,)))


num_steps = num_frames
for t in tqdm(range(start_t, start_t+num_steps)):
    gt_image = jnp.array(coord_images[t])
    for i in range(pose_estimates.shape[0]):
        enumerations_full = jnp.tile(jnp.eye(4)[None, :,:],(enumerations.shape[0], pose_estimates.shape[0],1,1))
        enumerations_full = enumerations_full.at[:,i,:,:].set(enumerations)
        proposals = jnp.einsum("bij,abjk->abik", pose_estimates, enumerations_full)

        weights = batched_scorer_parallel_jit(proposals, gt_image)
        pose_estimates = proposals[weights.argmax()]
    inferred_poses.append(pose_estimates.copy())


all_images = []
for t in range(start_t, start_t+num_steps):
    rgb_viz = Image.fromarray(
        rgb_images[t].astype(np.int8), mode="RGB"
    )
    gt_depth_1 = get_depth_image(coord_images[t][:,:,2], max=5.0)
    depth = render_planes_multiobject_jit(inferred_poses[t-start_t])
    depth = get_depth_image(depth[:,:,2], max=5.0)
    all_images.append(
        multi_panel([rgb_viz, gt_depth_1, depth], ["RGB Image", "Actual Depth", "Reconstructed Depth"], middle_width=10, top_border=100, fontsize=20)
    )

make_gif_from_pil_images(all_images, "swap_out_r01.gif")