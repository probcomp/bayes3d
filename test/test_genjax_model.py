import os

import genjax
import jax
import jax.numpy as jnp

import bayes3d as b
import bayes3d.genjax

key = jax.random.PRNGKey(1)

intrinsics = b.Intrinsics(
    height=100, width=100, fx=300.0, fy=300.0, cx=50.0, cy=50.0, near=0.01, far=20.0
)

b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj")
)
b.RENDERER.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj")
)

importance_jit = jax.jit(b.model.importance)

table_pose = b.t3d.inverse_pose(
    b.t3d.transform_from_pos_target_up(
        jnp.array([0.0, 4.8, 4.15]),
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
    )
)

enumerators = b.make_enumerator(["contact_params_1"])


def test_genjax_trace_contains_right_info():
    key = jax.random.PRNGKey(1)
    low, high = jnp.array([-0.2, -0.2, -jnp.pi]), jnp.array([0.2, 0.2, jnp.pi])
    weight, trace = importance_jit(
        key,
        genjax.choice_map(
            {
                "parent_0": -1,
                "parent_1": 0,
                "id_0": jnp.int32(1),
                "id_1": jnp.int32(0),
                "root_pose_0": table_pose,
                "camera_pose": jnp.eye(4),
                "face_parent_1": 3,
                "face_child_1": 2,
                "variance": 0.0001,
                "outlier_prob": 0.0001,
                "contact_params_1": jax.random.uniform(
                    key, shape=(3,), minval=low, maxval=high
                ),
            }
        ),
        (
            jnp.arange(2),
            jnp.arange(22),
            jnp.array([-jnp.ones(3) * 100.0, jnp.ones(3) * 100.0]),
            jnp.array(
                [
                    jnp.array([-0.5, -0.5, -2 * jnp.pi]),
                    jnp.array([0.5, 0.5, 2 * jnp.pi]),
                ]
            ),
            b.RENDERER.model_box_dims,
            1.0,
            intrinsics.fx,
        ),
    )

    scores = enumerators.enumerate_choices_get_scores(trace, key, jnp.zeros((100, 3)))

    assert trace["parent_0"] == -1
    assert (trace["camera_pose"] == jnp.eye(4)).all()
    assert trace["id_0"] == 0
