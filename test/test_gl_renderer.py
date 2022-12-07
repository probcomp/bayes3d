import os

from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.utils
import jax3dp3.gl_renderer
import jax3dp3.transforms_3d as t3d
import jax
import time
import trimesh
import numpy as np
import timeit


height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)

bunny_path = os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj")
mesh = trimesh.load(bunny_path)
trimesh_shape = (10.0*mesh.vertices)[mesh.faces] * jnp.array([1.0, -1.0, 1.0])
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))

gl_renderer = jax3dp3.gl_renderer.GLRenderer(height, width, fx, fy, cx, cy, 0.001, 100.0)
gl_renderer.load_vertices(np.array(trimesh_shape))

ITERS = 10
start = timeit.default_timer()
for _ in range(ITERS):
    img = gl_renderer.render([0],[pose])
end = timeit.default_timer()
print("Time: {}s".format((end - start)/ITERS))
save_depth_image(img[:,:,2], "bunny.png", max=6.0)

from IPython import embed; embed()

