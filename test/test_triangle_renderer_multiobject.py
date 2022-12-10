import os

from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.utils
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
from jax3dp3.triangle_renderer import ray_triangle_vmap, render_triangles_multiobject, ray_triangle_vmap_pose
import jax
import time
import trimesh
import functools

height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = mesh.vertices[mesh.faces]

trimesh_shape = jnp.concatenate([trimesh_shape, trimesh_shape])
trimesh_shape_h = jnp.concatenate([trimesh_shape, jnp.ones((*trimesh_shape.shape[:2],1))], axis=-1)

print('trimesh_shape.shape:');print(trimesh_shape.shape)
poses = jnp.stack([
    t3d.transform_from_pos(jnp.array([1.0, 1.0, 5.0])),
    t3d.transform_from_pos(jnp.array([-1.0, -1.0, 5.0]))
])
idxs = jnp.zeros((trimesh_shape.shape[0],),dtype=jnp.int8)
print('idxs.shape:');print(idxs.shape)

points = jax.vmap(ray_triangle_vmap_pose, in_axes=(None, 0, 0, None), out_axes=-2)(
    rays,
    trimesh_shape_h,
    idxs,
    poses
)
from IPython import embed; embed()

img = render_triangles_multiobject(poses, idxs, trimesh_shape_h, rays)
save_depth_image(img[:,:,2], "out.png", max=10.0)

from IPython import embed; embed()



from IPython import embed; embed()

img = jax3dp3.triangle_renderer.render_triangles(pose, trimesh_shape, rays)
save_depth_image(img[:,:,2], "triangle.png", max=6.0)


def f(a):
    return jnp.sum(a)

f_jit = jax.jit(f)
f_jit(jnp.ones((4,3)))

from IPython import embed; embed()

