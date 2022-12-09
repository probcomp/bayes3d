import sys
sys.path.append('.')
from functools import partial 
import jax.numpy as jnp
import jax
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.coarse_to_fine import coarse_to_fine
from jax3dp3.distributions import gaussian_vmf
from jax3dp3.enumerations import get_rotation_proposals
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.utils import make_centered_grid_enumeration_3d_points
from jax3dp3.viz import save_depth_image, get_depth_image
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.enumerations import make_translation_grid_enumeration
from PIL import Image
import time

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
gx, gy, gz = 0.0, 0.0, 1.50
gt_pose = jnp.array([
    [1.0, 0.0, 0.0, gx],   
    [0.0, 1.0, 0.0, gy],   
    [0.0, 0.0, 1.0, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)

f_jit = jax.jit(jax.vmap(lambda t: jnp.vstack(
                [jnp.hstack([jnp.eye(3), t.reshape(3,-1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]))
             )
pose_deltas = f_jit(make_centered_grid_enumeration_3d_points(0.4, 0.4, 0.5, 4, 4, 4))
tr_proposals = jnp.einsum("ij,ajk->aik", gt_pose, pose_deltas)


rot_proposals = get_rotation_proposals(8,8)
proposals = jnp.einsum("aij,bjk->abik", tr_proposals, rot_proposals).reshape(-1, 4,4)
    


# qualitatively testing the goodness of the test set of poses
def get_test_images(shape, shape_name):
    render_planes_lambda = lambda p, shape: render_planes(p,shape,h,w,fx_fy,cx_cy)
    render_planes_jit = jax.jit(render_planes_lambda)
    # render_planes_parallel_jit = jax.jit(jax.vmap(render_planes_lambda))

    default_img = render_planes_jit(gt_pose, shape)
    save_depth_image(default_img[:,:,2], f"test_default_{shape_name}.png", max=max_depth)


    images = []
    for i, pose in enumerate(proposals):
        print(i)
        img = render_planes_jit(pose, shape)
        depth = get_depth_image(img[:, :, 2], max=max_depth)
        images.append(depth)
        

    images[0].save(
        fp=f"test_{shape_name}.gif",
        format="GIF",
        append_images=images,
        save_all=True,
        duration=1,
        loop=0,
    )


######
# Rectangular prism 0: (cube)
######
cube_length = 0.5
shape = get_cube_shape(cube_length)
get_test_images(shape, "cube")


######
# Rectangular prism 1: (near-cube)
######
rect_shape = jnp.array([0.45, 0.5, 0.55])
shape = get_rectangular_prism_shape(rect_shape)
get_test_images(shape, "rect1")

######
# Rectangular prism 2: (2 similar dims, 1 unsimilar dim)
######
rect_shape = jnp.array([0.40, 0.42, 0.8])
shape = get_rectangular_prism_shape(rect_shape)
get_test_images(shape, "rect2")


######
# Rectangular prism 3: (3 unsimilar dims)
######
rect_shape = jnp.array([0.3, 0.5, 0.7])
shape = get_rectangular_prism_shape(rect_shape)
get_test_images(shape, "rect3")


######
# Rectangular prism 4: (near-(square)plane)
######
rect_shape = jnp.array([0.05, 0.5, 0.5])
shape = get_rectangular_prism_shape(rect_shape)
get_test_images(shape, "rect4")

