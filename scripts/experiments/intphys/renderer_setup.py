import bayes3d as b
import numpy as np
import jax.numpy as jnp
import jax
import os


def setup_renderer_and_meshes_v1(height, width, focal_length, near,
                               far, ids=range(1,22)):
    intrinsics = b.Intrinsics(
    height=height,
    width=width,
    fx=focal_length, fy=focal_length,
    cx=width/2.0, cy=height/2.0,
    near=near, far=far
    )

    b.setup_renderer(intrinsics)
    # loop through the b.utils.get_assets_dir() + "bop/ycbv/models" and add all the meshes
    model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
    for idx in ids:
        mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
        b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=1.0/1000.0, mesh_name = "obj_" + "{}".format(idx))

    # loop through the b.utils.get_assets_dir() + "sample_obs" and add all the meshes
    for file in os.listdir(os.path.join(b.utils.get_assets_dir(), "sample_objs")):  
        if file.endswith(".obj"):
            b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs", file), scaling_factor=0.1, mesh_name = file[:-4])

def setup_renderer_and_meshes_v2(height, width, focal_length, near,
                               far):
    intrinsics = b.Intrinsics(
    height=height,
    width=width,
    fx=focal_length, fy=focal_length,
    cx=width/2.0, cy=height/2.0,
    near=near, far=far
    )

    b.setup_renderer(intrinsics)

    # only add the bunny obj
    b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs/bunny.obj")
                                  , scaling_factor=0.1, mesh_name = "bunny")

