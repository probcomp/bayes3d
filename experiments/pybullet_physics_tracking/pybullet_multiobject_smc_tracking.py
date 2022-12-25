import numpy as np
import os
import jax.numpy as jnp
import jax
import trimesh
from jax3dp3.rendering import render_planes
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, depth_to_point_cloud_image
from jax3dp3.distributions import gaussian_vmf, gaussian_vmf_cov
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from jax.scipy.special import logsumexp
from jax3dp3.viz import multi_panel
from jax3dp3.enumerations import make_grid_enumeration, make_translation_grid_enumeration
from jax3dp3.rendering import render_spheres, render_cloud_at_pose,render_planes_multiobject
from jax3dp3.rendering import render_planes_multiobject
from jax3dp3.utils import get_assets_dir
import jax3dp3
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d


data = np.load("data.npz")
depth_imgs = np.array(data["depth_imgs"]).copy()
rgb_imgs = np.array(data["rgb_imgs"]).copy()


min_depth = 900.0
max_depth = 1200.0
middle_width = 20
top_border = 100
cm = plt.get_cmap("turbo")

scaling_factor = 0.25

fx = data["fx"] * scaling_factor
fy = data["fy"] * scaling_factor

cx = data["cx"] * scaling_factor
cy = data["cy"] * scaling_factor

original_height = data["height"]
original_width = data["width"]


h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)

coord_images = [depth_to_point_cloud_image(cv2.resize(d.copy(), (w,h),interpolation=1), fx,fy,cx,cy) for d in depth_imgs]
ground_truth_images = np.stack(coord_images)
ground_truth_images[ground_truth_images[:,:,:,2] > 1900.0] = 0.0
# ground_truth_images[ground_truth_images[:,:,:,1] > 0.85,:] = 0.0
ground_truth_images = np.concatenate([ground_truth_images, np.ones(ground_truth_images.shape[:3])[:,:,:,None] ], axis=-1)
ground_truth_images = jnp.array(ground_truth_images)

r = 1.0
outlier_prob = 0.05

shape_filenames = [
    os.path.join(jax3dp3.utils.get_assets_dir(), "models/003_cracker_box/textured_simple.obj"),
    os.path.join(jax3dp3.utils.get_assets_dir(), "models/004_sugar_box/textured_simple.obj")
]
shape_planes = []
shape_dims = []
for f in shape_filenames:
    mesh = trimesh.load(f)
    half_dims = np.array(mesh.vertices).max(0)
    print(half_dims)
    plane, dims = shape = get_rectangular_prism_shape(half_dims*2.0)
    shape_planes.append(plane)
    shape_dims.append(dims)

shape_dims = jnp.array(shape_dims)
shape_planes = jnp.array(shape_planes)

render_from_pose = lambda pose: render_planes_multiobject(pose, shape_planes, shape_dims, h,w, fx,fy, cx,cy)
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


cam_pose = np.eye(4)
cam_pose[:3,0] = np.array([0.0, 1.0, 0.0])
cam_pose[:3,1] = np.array([0.0, 0.0, -1.0])
cam_pose[:3,2] = np.array([-1.0, 0.0, 0.0])
cam_pose[:3,3] = np.array([1000.0, 0.0, 0.0])
cam_pose = jnp.array(cam_pose)

initial_poses_estimates = jnp.array(
    [
        [
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, -30.00],
            [0.0, 1.0, 0.0, -200.00],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ]
)
initial_poses_estimates = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), initial_poses_estimates)

rgb = rgb_imgs[0]
rgb_img = Image.fromarray(
    rgb.astype(np.int8), mode="RGBA"
)
rgb_img.save("rgb.png")

initial_poses = initial_poses_estimates.copy()
rendered_image = render_from_pose_jit(initial_poses)
before = jax3dp3.viz.get_depth_image(rendered_image[:, :, 2],min=min_depth,max=max_depth).resize((original_width,original_height))


# for _ in range(10):
#     for i in range(initial_poses.shape[0]):
#         enumerations1 = make_grid_enumeration(-40.0, -40.0, -40.0, 40.0, 40.0, 40.0, 10, 10, 10, 2, 1)
#         initial_poses_expanded = jnp.tile(initial_poses[None, :, :, :], (enumerations1.shape[0], 1, 1, 1))
#         proposals = initial_poses_expanded.at[:, i].set(jnp.einsum("...ij,...jk->...ik", initial_poses_expanded[:, i], enumerations1))
#         weights = likelihood_parallel_jit(proposals, ground_truth_images[0])
#         initial_poses = proposals[jnp.argmax(weights)]

save_depth_image(ground_truth_images[0][:,:,2],"gt_depth.png",min=min_depth,max=max_depth)
save_depth_image(render_from_pose(initial_poses)[:,:,2],"inferred_depth.png",min=min_depth,max=max_depth)

rendered_image = render_from_pose_jit(initial_poses)
after = Image.fromarray(
    (cm(np.array(rendered_image[:, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
).resize((original_width,original_height))
dst = multi_panel([rgb_img, before, after], ["rgb", "before","after"], middle_width, top_border, 40)
dst.save("before_after.png")

num_grids = 1000
idx = 50
i = 1
proposals = jnp.array([initial_poses for _ in range(num_grids)])
proposals = proposals.at[:,i,0,3].set(jnp.linspace(-200.0, 200.0, num_grids))
scores = likelihood_parallel(proposals, ground_truth_images[idx])
best_poses = proposals[scores.argmax()]

rgb = rgb_imgs[idx]
rgb_img = Image.fromarray(
    rgb.astype(np.int8), mode="RGBA"
)

depth_img = get_depth_image(ground_truth_images[idx,:,:,2],min=min_depth,max=max_depth).resize((original_width,original_height))
rendered_image = render_from_pose_jit(best_poses)
rendered_depth_img = get_depth_image(rendered_image[:, :, 2],min=min_depth,max=max_depth).resize((original_width,original_height))

i1 = rendered_depth_img.copy()
i2 = rgb_img.copy()
i1.putalpha(128)
i2.putalpha(128)
overlay_img = Image.alpha_composite(i1, i2)

dst = multi_panel([rgb_img, depth_img, rendered_depth_img, overlay_img], None, middle_width, top_border, 40)
dst.save("best.png")

num_particles = 1000
drift_poses = make_translation_grid_enumeration(-14.0,0.0,0.0,14.0,0.0,0.0,1000,1,1)


def run_inference(initial_particles, ground_truth_images):
    def particle_filtering_step(data, gt_image):
        particles, weights, keys = data
        i = 1
        particles = particles.at[:, i].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[:,i]))
        scores = likelihood_parallel(particles, gt_image)
        weights = weights + scores
        # parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
        particles = jnp.tile(particles[scores.argmax()][None,...],(weights.shape[0],1,1,1))
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))
        keys = jax.random.split(keys[0], weights.shape[0])
        return (particles, weights, keys), particles

    initial_weights = jnp.full(initial_particles.shape[0], 0.0)
    initial_key = jax.random.PRNGKey(3)
    initial_keys = jax.random.split(initial_key, initial_particles.shape[0])
    return jax.lax.scan(particle_filtering_step, (initial_particles, initial_weights, initial_keys), ground_truth_images)


run_inference_jit = jax.jit(run_inference)
particles = []
for _ in range(num_particles):
    particles.append(initial_poses)
particles = jnp.stack(particles)

_,x = run_inference_jit(particles, ground_truth_images)


start = time.time()
_,x = run_inference_jit(particles, ground_truth_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", ground_truth_images.shape[0] / (end - start))




middle_width = 20
top_border = 100
cm = plt.get_cmap("turbo")
all_images = []
for i in range(ground_truth_images.shape[0]):
    rgb = rgb_imgs[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )

    depth_img = get_depth_image(ground_truth_images[i,:,:,2],min=min_depth,max=max_depth).resize((original_width,original_height))

    pose = x[i,0,:,:,:]
    rendered_image = render_from_pose_jit(pose)

    rendered_depth_img = get_depth_image(rendered_image[:, :, 2],min=min_depth,max=max_depth).resize((original_width,original_height))

    i1 = rendered_depth_img.copy()
    i2 = rgb_img.copy()
    i1.putalpha(128)
    i2.putalpha(128)
    overlay_img = Image.alpha_composite(i1, i2)

    translations = x[i, :, 1, :3, -1]
    img = render_cloud_at_pose(translations, jnp.eye(4), h, w, fx,fy, cx,cy, 0)
    projected_particles_img = get_depth_image(img[:,:,2], max=40.0).resize((original_width,original_height))



    images = [rgb_img, depth_img, rendered_depth_img, overlay_img, projected_particles_img]
    labels = ["RGB Image Frame {}".format(i), "Depth Image", "Inferred Depth", "Overlay",  "Particles"]
    dst = multi_panel(images, labels, middle_width, top_border, 40)
    all_images.append(dst)


all_images[0].save(
    fp="out_{}_particles.gif".format(num_particles),
    format="GIF",
    append_images=all_images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()