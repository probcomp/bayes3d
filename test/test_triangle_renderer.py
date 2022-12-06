import os

from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.utils
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
import trimesh


height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = (10.0*mesh.vertices)[mesh.faces] * jnp.array([1.0, -1.0, 1.0])
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))
img = jax3dp3.triangle_renderer.render_triangles(pose, trimesh_shape, rays)
save_depth_image(img[:,:,2], "triangle.png", max=6.0)

from IPython import embed; embed()

