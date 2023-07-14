import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time
import torch
import bayes3d._rendering.nvdiffrast.common as dr


intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
b.setup_renderer(intrinsics)
renderer = b.RENDERER

r = 0.1
outlier_prob = 0.01
max_depth = 15.0

renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))
renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/sphere.obj"))

poses = jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None, None, ...]
indices = jnp.array([0])


img = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(dr._get_plugin(gl=True).rasterize_fwd_gl(
    b.RENDERER.renderer_env.cpp_wrapper, 
    torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.tile(poses, (1,2,1,1)))),
    b.RENDERER.proj_list,
    [0]
)))
b.get_depth_image(img[0,:,:,2]).save("1.png")
assert not jnp.all(img[0,:,:,2] == 0.0)

multiobject_scene_img = renderer._render_many(jnp.tile(poses, (2,1,1,1)), jnp.array([1]))[0]
b.get_depth_image(multiobject_scene_img[:,:,2]).save("0.png")