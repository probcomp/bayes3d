import os

import jax
import jax.numpy as jnp

import bayes3d as b

are_bboxes_intersecting_jit = jax.jit(b.utils.are_bboxes_intersecting)

# set up renderer
intrinsics = b.Intrinsics(
    height=100, width=100, fx=250, fy=250, cx=100 / 2.0, cy=100 / 2.0, near=0.1, far=20
)

b.setup_renderer(intrinsics)

b.RENDERER.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"),
    scaling_factor=0.1,
    mesh_name="cube_1",
)
b.RENDERER.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"),
    scaling_factor=0.1,
    mesh_name="cube_2",
)

# make poses intersect/collide/penetrate
pose_1 = jnp.eye(4).at[:3, 3].set([-0.1, 0, 1.5])
pose_1 = pose_1 @ b.transform_from_axis_angle(jnp.array([1, 0, 0]), jnp.pi / 4)
pose_2 = jnp.eye(4).at[:3, 3].set([-0.05, 0, 1.5])
pose_2 = pose_2 @ b.transform_from_axis_angle(jnp.array([1, 1, 1]), jnp.pi / 4)

# make sure the output confirms the intersection
b.scale_image(
    b.get_depth_image(
        b.RENDERER.render(jnp.stack([pose_1, pose_2]), jnp.array([0, 1]))[:, :, 2]
    ),
    4,
).save("intersecting.png")
is_intersecting = are_bboxes_intersecting_jit(
    b.RENDERER.model_box_dims[0], b.RENDERER.model_box_dims[1], pose_1, pose_2
)
assert is_intersecting is True

# make poses NOT intersect/collided/penetrate
pose_2 = jnp.eye(4).at[:3, 3].set([0.04, 0, 1.5])
pose_2 = pose_2 @ b.transform_from_axis_angle(jnp.array([1, 1, 1]), jnp.pi / 4)

# make sure the output confirms NO intersection
b.scale_image(
    b.get_depth_image(
        b.RENDERER.render(jnp.stack([pose_1, pose_2]), jnp.array([0, 1]))[:, :, 2]
    ),
    4,
).save("no_intersecting.png")
is_intersecting = are_bboxes_intersecting_jit(
    b.RENDERER.model_box_dims[0], b.RENDERER.model_box_dims[1], pose_1, pose_2
)
assert is_intersecting is False
