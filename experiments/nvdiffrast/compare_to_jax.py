import os

from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.utils
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
import jax
import time
import trimesh

height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = mesh.vertices[mesh.faces]
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))
num_images = 1
poses_all = jnp.array([pose for _ in range(num_images)])
parallel_renderer = jax.jit(jax.vmap(lambda pose: jax3dp3.triangle_renderer.render_triangles(pose, trimesh_shape, rays)))
parallel_renderer(poses_all)

x = parallel_renderer(poses_all)
x = parallel_renderer(poses_all)
x = parallel_renderer(poses_all)
new_poses = poses_all + 0.001
start = time.time()
x = parallel_renderer(new_poses)
print(len(x))
end = time.time()
print ("Time elapsed:", end - start)

