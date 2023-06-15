import jax.numpy as jnp
import bayes3d as b
import trimesh
import os


mesh_names = ["bunny", "sphere", "pyramid", "cube"]

meshes = [trimesh.load(os.path.join(b.utils.get_assets_dir(), f"sample_objs/{name}.obj")) for name in mesh_names]
meshes = [b.mesh.center_mesh(mesh) for mesh in meshes]
mesh_dims = [b.utils.aabb(mesh.vertices)[0] for mesh in meshes]

table_dims = jnp.array([10.0, 10.0, 0.1])
meshes = [b.mesh.make_cuboid_mesh(table_dims), *meshes]

intrinsics = b.Intrinsics(
    200,
    200,
    400.0,
    400.0,
    100.0,
    100.0,
    0.02,
    30.0
)
b.setup_renderer(intrinsics)
for mesh in meshes:
    b.RENDERER.add_mesh(mesh)

box_dims = [b.utils.aabb(m.vertices)[0] for m in meshes]
box_dims = jnp.array(box_dims)

absolute_poses = jnp.array([
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4),
])

parents = jnp.array([
    -1, 0, 0, 0, 0
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

poses = b.scene_graph.poses_from_scene_graph_jit(
    absolute_poses, box_dims, parents, contact_params, face_parents, face_child
)

pos = jnp.array([[10.0, -15.0, 5.0]])
target = jnp.array([0.0, 0.0, 0.0])
up = jnp.array([0.0, 0.0, 1.0])

camera_pose = b.t3d.transform_from_pos_target_up(pos, target, up)



poses_in_camera_frame = b.t3d.inverse_pose(camera_pose) @ poses

img = b.RENDERER.render_multiobject(b.t3d.inverse_pose(camera_pose) @ poses, jnp.arange(len(poses)))
b.get_depth_image(img[:,:,2]).save("test_scene_graph.png")

from IPython import embed; embed()
