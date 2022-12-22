import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_planes, render_sphere, render_cloud_at_pose
from jax3dp3.distributions import gaussian_vmf
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix
from jax3dp3.enumerations import make_translation_grid_enumeration
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from jax3dp3.likelihood import threedp3_likelihood, sample_cloud_within_r
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
import matplotlib.pyplot as plt

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
fx_fy = jnp.array([fx,fy])
cx_cy = jnp.array([cx,cy])

outlier_prob = 0.05
max_depth = 8.0

pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))

cube_shape = get_cube_shape(1.0)
sphere_obs_image = render_sphere(pose, 0.5, h,w, fx,fy, cx,cy)
cube_obs_image = render_planes(pose, cube_shape, h,w, fx,fy, cx,cy)
sphere_viz =jax3dp3.viz.get_depth_image(sphere_obs_image[:,:,2], max=max_depth)
cube_viz = jax3dp3.viz.get_depth_image(cube_obs_image[:,:,2],  max=max_depth)

jax3dp3.viz.multi_panel(
    [sphere_viz, cube_viz, cube_viz],
    ["Sphere Observation", "Cube Observation", "Latent Scene"]
).save("data.png")



def scorer(r, obs):
    rendered_image = render_planes(pose, cube_shape, h,w, fx,fy, cx,cy)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes = (0, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

r_sweep = jnp.linspace(0.0, 1.0, 5000)
sphere_r_posterior = scorer_parallel_jit(r_sweep, sphere_obs_image)
cube_r_posterior = scorer_parallel_jit(r_sweep, cube_obs_image)

plt.clf()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
fig.suptitle("Posterior over likelihood radius")
ax1.plot(r_sweep, jnp.exp(sphere_r_posterior - sphere_r_posterior.max()))
ax1.set_title("for sphere observation")

ax2.plot(r_sweep, jnp.exp(cube_r_posterior - cube_r_posterior.max()))
ax2.set_title("for cube observation")
plt.savefig("r_posteriors.png")

key = jax.random.PRNGKey(3)
pixel_smudge = 0

best_r_sphere = r_sweep[sphere_r_posterior.argmax()]
sampled_cloud_r = sample_cloud_within_r(key, sphere_obs_image[:,:,:3], best_r_sphere, duplicates=10)
rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx,fy, cx,cy, pixel_smudge)
sphere_noise_viz = jax3dp3.viz.get_depth_image(rendered_cloud_r[:,:,2],  max=max_depth)

best_r_cube = r_sweep[cube_r_posterior.argmax()]
sampled_cloud_r = sample_cloud_within_r(key, cube_obs_image[:,:,:3], best_r_cube, duplicates=10)
rendered_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, fx,fy, cx,cy, pixel_smudge)
cube_noise_viz = jax3dp3.viz.get_depth_image(rendered_cloud_r[:,:,2],  max=max_depth)


jax3dp3.viz.multi_panel(
    [sphere_noise_viz, cube_noise_viz],
    ["Sphere At Inferred Noise Level", "Cube At Inferred Noise Level"],
    middle_width=100,

).save("noise.png")







from IPython import embed; embed()