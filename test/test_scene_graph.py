import jax.numpy as jnp
import bayes3d as j
import trimesh
import os


mesh_names = ["bunny", "sphere", "pyramid", "cube"]

meshes = [trimesh.load(os.path.join(j.utils.get_assets_dir(), f"sample_objs/{name}.obj")) for name in mesh_names]
meshes = [j.mesh.center_mesh(mesh) for mesh in meshes]
mesh_dims = [j.utils.aabb(mesh.vertices)[0] for mesh in meshes]

table_dims = jnp.array([10.0, 10.0, 0.1])
meshes = [j.mesh.make_cuboid_mesh(table_dims), *meshes]

intrinsics = j.Intrinsics(
    200,
    400,
    400.0,
    400.0,
    200.0,
    100.0,
    0.02,
    30.0
)
renderer = j.Renderer(intrinsics)
for mesh in meshes:
    renderer.add_mesh(mesh)

box_dims = [j.utils.aabb(m.vertices)[0] for m in meshes]
box_dims = jnp.array(box_dims)

absolute_poses = jnp.array([
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
])

edges = jnp.array([
    [-1,0],
    [0,1],
    [0,2],
    [0,3],
    [0,4],
])

contact_params = jnp.array(
    [
        [0.0, 0.0, jnp.pi/4],
        [-1.0, -0.5, jnp.pi/4],
        [-0.2, 0.1, jnp.pi/2],
        [2.0, -1.0, jnp.pi/2],
        [1.0, -4.0, jnp.pi/2],
    ]
)

face_parents = jnp.array([2,2,2,2,2])
face_child = jnp.array([0,0,0,0,0])

poses = j.scene_graph.poses_from_scene_graph_jit(
    absolute_poses, box_dims, edges, contact_params, face_parents, face_child
)

pos = jnp.array([[10.0, -15.0, 5.0]])
target = jnp.array([0.0, 0.0, 0.0])
up = jnp.array([0.0, 0.0, 1.0])

camera_pose = j.t3d.transform_from_pos_target_up(pos, target, up)



poses_in_camera_frame = j.t3d.inverse_pose(camera_pose) @ poses

img = renderer.render_multiobject(j.t3d.inverse_pose(camera_pose) @ poses, jnp.arange(len(poses)))
j.get_depth_image(img[:,:,2]).save("test_scene_graph.png")

from IPython import embed; embed()
