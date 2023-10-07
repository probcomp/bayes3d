import bayes3d as b
import genjax
import jax.numpy as jnp
import jax
import os
import matplotlib.pyplot as plt
import jax.tree_util as jtu
from tqdm import tqdm
import bayes3d.genjax
console = genjax.pretty(show_locals=False)
from genjax._src.core.transforms.incremental import NoChange
from genjax._src.core.transforms.incremental import UnknownChange
from genjax._src.core.transforms.incremental import Diff
import inspect
import joblib

intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=500.0, fy=500.0,
    cx=50.0, cy=50.0,
    near=0.01, far=20.0
)

b.setup_renderer(intrinsics)
model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
meshes = []
for idx in range(1,22):
    mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
    b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=1.0/1000.0)

b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"), scaling_factor=1.0/1000000000.0)

table_pose = b.t3d.inverse_pose(
    b.t3d.transform_from_pos_target_up(
        jnp.array([0.0, 2.0, 1.20]),
        jnp.array([0.0, 0.0, 0.0]),
        jnp.array([0.0, 0.0, 1.0]),
    )
)

OUTLIER_VOLUME = 100.0
key = jax.random.PRNGKey(500)

importance_jit = jax.jit(b.genjax.model.importance)

scene_id = 0
while True:
    if scene_id >= 100:
        break
    key, (_,trace) = importance_jit(key, genjax.choice_map({
        "parent_0": -1,
        "parent_1": 0,
        "parent_2": 0,
        "parent_3": 0,
        "id_0": jnp.int32(21),
        "camera_pose": jnp.eye(4),
        "root_pose_0": table_pose,
        "face_parent_1": 2,
        "face_parent_2": 2,
        "face_parent_3": 2,
        "face_child_1": 3,
        "face_child_2": 3,
        "face_child_3": 3,
    }), (
        jnp.arange(4),
        jnp.arange(22),
        jnp.array([-jnp.ones(3)*100.0, jnp.ones(3)*100.0]),
        jnp.array([jnp.array([-0.2, -0.2, -2*jnp.pi]), jnp.array([0.2, 0.2, 2*jnp.pi])]),
        b.RENDERER.model_box_dims, OUTLIER_VOLUME)
    )
    if (b.genjax.get_indices(trace) == 21).sum() > 1:
        continue
    
    joblib.dump((trace.get_choices(), trace.get_args()), f"data/trace_{scene_id}.joblib")
    scene_id += 1