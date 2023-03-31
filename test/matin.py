import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
j.meshcat.setup_visualizer()


mesh_names = ["bunny", "sphere", "pyramid", "cube"]

bunny_mesh = trimesh.load(os.path.join(j.utils.get_assets_dir(), f"sample_objs/bunny.obj"))
bunny_mesh = j.mesh.center_mesh(bunny_mesh)
bunny_dims, bunny_center_pose = j.utils.aabb(bunny_mesh.vertices)

table_dims = jnp.array([27.0, 0.1, 20.0])
table_mesh = j.mesh.make_cuboid_mesh(table_dims)

intrinsics = j.Intrinsics(
    200,
    200,
    400.0,
    400.0,
    200.0,
    100.0,
    0.02,
    50.0
)
renderer = j.Renderer(intrinsics)
renderer.add_mesh(table_mesh)
renderer.add_mesh(bunny_mesh)

absolute_poses = jnp.array([
    jnp.eye(4),
    jnp.eye(4),
])

edges = jnp.array([
    [-1,0],
    [0,1],
])

contact_params = jnp.array(
    [
        [0.0, 0.0, jnp.pi/4],
        [1.0, 1.0, -jnp.pi/4],
    ]
)

face_parents = jnp.array([1,1])
face_child = jnp.array([0,0])

poses = j.scene_graph.absolute_poses_from_scene_graph_jit(
    absolute_poses, renderer.model_box_dims, edges, contact_params, face_parents, face_child
)


j.meshcat.clear()
j.meshcat.show_trimesh("table", table_mesh, color=j.BLUE)
j.meshcat.set_pose("table", poses[0])

j.meshcat.show_trimesh("bunny", bunny_mesh)
j.meshcat.set_pose("bunny", poses[1])

pos = jnp.array([[0.0, -7.0, 7.0]])
target = jnp.array([0.0, 0.0, 0.0])
up = jnp.array([0.0, -1.0, 0.0])

camera_pose = j.t3d.transform_from_pos_target_up(pos, target, up)

img = renderer.render_multiobject(
    jnp.linalg.inv(camera_pose) @ poses,
    [0,1]
)
j.get_depth_image(img[:,:,2],max=10.0).save("bunny_table.png")

