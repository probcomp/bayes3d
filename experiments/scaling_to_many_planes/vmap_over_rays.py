import jax.numpy as jnp
import jax
import functools
from jax3dp3.viz import save_depth_image
from jax3dp3.triangle_renderer import render_triangles
from jax3dp3.transforms_3d import transform_from_pos
import time


image_size = (100,100)
height, width = image_size
fx = 100.0
fy = 100.0
cx = (width-1)/2
cy = (height-1)/2

rows, cols = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
pixel_coords = jnp.stack([rows,cols],axis=-1)
pixel_coords_dir = jnp.concatenate([(pixel_coords - jnp.array([cx,cy])) / jnp.array([fx,fy]), jnp.ones((height,width,1))],axis=-1)

rays = pixel_coords_dir

import trimesh
# mesh = trimesh.load("icosahedron.obj")
# trimesh_shape = (2.0*mesh.vertices)[mesh.faces]

mesh = trimesh.load("bunny.obj")
trimesh_shape = (10.0*mesh.vertices)[mesh.faces] * jnp.array([1.0, -1.0, 1.0])
pose = transform_from_pos(jnp.array([0.0, 0.0, 6.0]))


poses = jnp.array([
    pose
    for _ in range(2)
])
render_triangles_parallel_jit = jax.jit(jax.vmap(render_triangles, in_axes=(0, None, None)))

points = render_triangles(
    pose,
    trimesh_shape,
    rays
)

start = time.time()
points = render_triangles_parallel_jit(
    poses,
    trimesh_shape + 0.2,
    rays
)
end = time.time()
print ("Time elapsed:", end - start)
save_depth_image(points[0][:,:,2], "test.png", max = 6.0)








from IPython import embed; embed()