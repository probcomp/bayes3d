import sys
sys.path.append('.')

import jax.numpy as jnp
import jax
from jax3dp3.coarse_to_fine import coarse_to_fine
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.shape import get_rectangular_prism_shape
from jax3dp3.viz import save_depth_image, get_depth_image
import time
from functools import partial 
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.enumerations import make_translation_grid_enumeration

## Camera settings
h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
fx, fy = fx_fy
cx, cy = cx_cy   
max_depth = 5.0
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

### Generate GT images
gx, gy, gz = 0.531, 0.251, 1.950
gt_pose = jnp.array([
    [0.9860675,  -0.16779144, -0.04418374, gx],   
    [0.17300624,  0.92314297,  0.33919233, gy],   
    [-0.01606147, -0.34134597,  0.94141835, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

shape_dims = jnp.array([0.3, 0.5, 0.7])  # shape to evaluate poses on
shape = get_rectangular_prism_shape(shape_dims)  

# create gt image from some shape and pose
gt_shape = get_rectangular_prism_shape(jnp.array([0.5, 0.5, 0.5]))  
render_planes_lambda = lambda p: render_planes(p,gt_shape,h,w,fx,fy,cx,cy)
render_planes_jit = jax.jit(render_planes_lambda)
render_planes_parallel_jit = jax.jit(jax.vmap(render_planes_lambda))
gt_image = render_planes_jit(gt_pose)


def scorer(pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx, fy, cx, cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight

print("GT image shape=", gt_image.shape)
print("GT pose=", gt_pose)

gt_depth_img = get_depth_image(gt_image[:,:,2], max_depth)
save_depth_image(gt_image[:,:,2], "gt_img.png", max=max_depth)

latent_pose_estimate = jnp.array([
    [1.0, 0.0, 0.0, 0.5],   
    [0.0, 1.0, 0.0, 0.25],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ])

print("initial latent=", latent_pose_estimate)

# tuples of (radius, width of gridding, num gridpoints)
schedule_tr = [(0.5, 1, 10), (0.25, 0.5, 10), (0.1, 0.2, 10), (0.02, 0.1, 10)]
schedule_rot = [(10, 10), (10, 10), (20, 20), (30,30)]

enumeration_likelihood_r = [sched[0] for sched in schedule_tr]
enumeration_grid_tr = [make_translation_grid_enumeration(
                        -grid_width, -grid_width, -grid_width,
                        grid_width, grid_width, grid_width,
                        num_grid_points,num_grid_points,num_grid_points
                    ) for (_, grid_width, num_grid_points) in schedule_tr]
enumeration_grid_r = [get_rotation_proposals(fib_nums, rot_nums) for (fib_nums, rot_nums) in schedule_rot]


coarse_to_fine = partial(coarse_to_fine, scorer)
coarse_to_fine_jit = jax.jit(coarse_to_fine)

_ = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r,
                        jnp.array([
                        [1.0, 0.0, 0.0, 0.0],   
                        [0.0, 1.0, 0.0, 0.0],   
                        [0.0, 0.0, 1.0, 0.0],   
                        [0.0, 0.0, 0.0, 0.0],   
                        ]), jnp.zeros(gt_image.shape))

start = time.time()                       
best_pose_estimate = coarse_to_fine_jit(enumeration_grid_tr, enumeration_grid_r, enumeration_likelihood_r, latent_pose_estimate, gt_image)             
end = time.time()
print(best_pose_estimate)
elapsed = end - start
print("time elapsed = ", elapsed, " FPS=", 1/elapsed)
best_img = render_planes_jit(best_pose_estimate)
save_depth_image(best_img[:,:,2], f"c2f_out.png",max=max_depth)

from IPython import embed; embed()
