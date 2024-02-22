import viser
import random
import time

import imageio.v3 as iio
import numpy as onp

server.add_frame(
    "/tree",
    wxyz=(1.0, 0.0, 0.0, 0.0),
    position=(random.random() * 2.0, 2.0, 0.2),
)
server.add_frame(
    "/tree/branch",
    wxyz=(1.0, 0.0, 0.0, 0.0),
    position=(random.random() * 2.0, 2.0, 0.2),
)

client_handle = list(server.get_clients().values())[0]

p,q = client_handle.camera.position, client_handle.camera.wxyz

client_handle.camera.position = p
client_handle.camera.wxyz = q

img = client_handle.camera.get_render(100,100)



server = viser.ViserServer()

import os
import trimesh
i = 9
model_dir = os.path.join(b.utils.get_assets_dir(), "ycb_video_models/models")
mesh_path = os.path.join(model_dir, b.utils.ycb_loader.MODEL_NAMES[i],"textured.obj")
mesh = trimesh.load(mesh_path)

server.add_mesh_trimesh(
    name="/trimesh",
    mesh=mesh,
)

server.reset_scene()


server.add_mesh(
    name="/trimesh",
    vertices=mesh.vertices,
    faces=mesh.faces,
)

sphere = trimesh.creation.uv_sphere(0.1, (10,10,))
server.add_mesh(
    name="/trimesh2",
    vertices=sphere.vertices * np.array([1.0, 2.0, 3.0]),
    faces=sphere.faces,
)