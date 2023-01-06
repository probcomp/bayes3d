import os
import numpy as np
import matplotlib.pyplot as plt
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax
import jax3dp3
from scipy.spatial.transform import Rotation as R
import jax3dp3.jax_rendering
from PIL import Image

from tqdm import tqdm
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


K = jnp.array([
    [fx_fy[0], 0.0, cx_cy[0]],
    [0.0, fx_fy[1], cx_cy[1]],
    [0.0, 0.0, 1.0],
])

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

coord_images = []
seg_images = []

for frame_idx in range(num_frames):
    # frame_idx = 20
    k = 5 if 5 <= frame_idx < 19 else 4 # 4 objects in frames [5:19]

    coord_image,_ = t3d.depth_to_coords_in_camera(depth_images[frame_idx], K)
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




start_t = 10
shape_planes, shape_dims, init_poses = [], [], []
seg_img = seg_images[start_t][:,:,2]

imgs = []
for obj_id in jnp.unique(seg_img):
    imgs.append(jax3dp3.viz.get_depth_image(seg_img == obj_id))
jax3dp3.viz.multi_panel(imgs, [], 10, 10, 13).save("masks.png")

for obj_id in jnp.unique(seg_img):
    if obj_id == 0:
        continue
    obj_mask = (seg_img == obj_id)
    
    object_points = coord_images[start_t][obj_mask]
    maxs = np.max(object_points,axis=0)
    mins = np.min(object_points,axis=0)
    dims = (maxs - mins)
    center_of_box = (maxs+mins)/2
    
    init_pose = t3d.transform_from_pos(center_of_box)
    init_poses.append(init_pose)

    shape, dim = jax3dp3.jax_rendering.get_rectangular_prism_shape(dims)
    # print('dims:');print(dims)
    # print('center_of_box:');print(center_of_box)
    # print("\n")
    shape_planes.append(shape)
    shape_dims.append(dim)
    # break

shape_planes = jnp.stack(shape_planes)
shape_dims = jnp.stack(shape_dims)
init_poses = jnp.stack(init_poses)

reconstruction_image = jax3dp3.jax_rendering.render_planes_multiobject(
    init_poses,
    shape_planes,
    shape_dims,
    height, width, fx,fy, cx, cy
)

def render_planes_multiobject_lambda(poses):
    return (
        jax3dp3.jax_rendering.render_planes_multiobject(poses, shape_planes, shape_dims, height, width, fx,fy, cx,cy)
    )
render_planes_multiobject_jit = jax.jit(render_planes_multiobject_lambda)

reconstruction_image = render_planes_multiobject_jit(init_poses)
jax3dp3.viz.save_depth_image(reconstruction_image[:,:,2], "reconstruction.png", max=5.0)


# reconstruction_image = render_planes_multiobject_jit(init_poses)



r = 0.005
outlier_prob = 0.01
def likelihood(x, params):
    obs= params[0]
    rendered_image = jax3dp3.jax_rendering.render_planes_multiobject(x, shape_planes, shape_dims, height, width, fx,fy, cx,cy)
    weight = jax3dp3.threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight

likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)



enumerations = jax3dp3.make_translation_grid_enumeration(-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 9, 9, 9)

cm = plt.get_cmap("turbo")
max_depth = 30.0
middle_width = 20
top_border = 100


pose_estimates = init_poses.copy()

inferred_poses = []
batched_scorer_parallel_jit = jax.jit(lambda poses, image: jax3dp3.jax_rendering.batched_scorer_parallel_params(likelihood_parallel, 9, poses, (image,)))

num_steps = 2
for t in tqdm(range(start_t, start_t+num_steps)):
    gt_image = jnp.array(coord_images[t])
    for i in range(pose_estimates.shape[0]):
        enumerations_full = jnp.tile(jnp.eye(4)[None, :,:],(enumerations.shape[0], pose_estimates.shape[0],1,1))
        enumerations_full = enumerations_full.at[:,i,:,:].set(enumerations)
        proposals = jnp.einsum("bij,abjk->abik", pose_estimates, enumerations_full)


        # print(proposals.shape)
        weights = batched_scorer_parallel_jit(proposals, gt_image)
        # print(weights.max())
        pose_estimates = proposals[weights.argmax()]
    inferred_poses.append(pose_estimates.copy())


all_images = []
for t in range(start_t, start_t+num_steps):
    rgb_viz = Image.fromarray(
        rgb_images[t].astype(np.int8), mode="RGB"
    )
    gt_depth_1 = jax3dp3.viz.get_depth_image(coord_images[t][:,:,2], max=5.0)
    depth = render_planes_multiobject_jit(inferred_poses[t-start_t])
    depth = jax3dp3.viz.get_depth_image(depth[:,:,2], max=5.0)
    all_images.append(
        jax3dp3.viz.multi_panel([rgb_viz, gt_depth_1, depth], ["RGB Image", "Actual Depth", "Reconstructed Depth"], middle_width=10, top_border=100, fontsize=20)
    )

jax3dp3.viz.make_gif_from_pil_images(all_images, "out.gif")


from IPython import embed; embed()