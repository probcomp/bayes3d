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
import timeit
from jax3dp3.likelihood import threedp3_likelihood


image_size_sweep = [16, 32, 64, 128, 256, 512]
num_triangles_sweep = [10,100,1000,10000]

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = (10.0*mesh.vertices)[mesh.faces] * jnp.array([1.0, -1.0, 1.0])
trimesh_shape = jnp.tile(trimesh_shape,(5,1,1))
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))
ITERS = 100

for image_size in image_size_sweep:
    height, width = image_size, image_size
    cx = (width-1)/2
    cy = (height-1)/2
    fx = 200.0
    fy = 200.0
    rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)
    shape = trimesh_shape[:10]
    render_func = jax.jit(lambda pose: jax3dp3.triangle_renderer.render_triangles(pose, shape, rays))
    img = render_func(pose)

    likelihood_jit = jax.jit(lambda img_in: threedp3_likelihood(img_in, img, 0.1, 0.0001))
    likelihood_jit(img)
    likelihood_jit(img)
    likelihood_jit(img)

    start = timeit.default_timer()
    for _ in range(ITERS):
        likelihood = likelihood_jit(img)
    end = timeit.default_timer()
    print("{}x{} Likelihood Evaluation Time: {}s".format(height, width, (end - start)/ITERS))


print("Sweeping image size and number of triangles")

for image_size in image_size_sweep:
    for num_triangles in num_triangles_sweep:
        height, width = image_size, image_size
        cx = (width-1)/2
        cy = (height-1)/2
        fx = 200.0
        fy = 200.0
        rays = jax3dp3.camera.camera_rays_from_params(height, width, fx, fy, cx, cy)
        f = jax.jit(lambda shape: jax3dp3.triangle_renderer.render_triangles(pose, shape, rays))

        mesh_sized = trimesh_shape[:num_triangles]
        img = f(mesh_sized)
        img = f(mesh_sized)

        start = timeit.default_timer()
        for _ in range(ITERS):
            img = f(mesh_sized + 0.002)
        end = timeit.default_timer()
        print ("{}x{} {} triangles, Rendering Time: {}s".format(height,width, mesh_sized.shape[0], (end - start)/ITERS))
