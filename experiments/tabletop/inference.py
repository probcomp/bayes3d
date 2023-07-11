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

VARIANCE_GRID = jnp.array([0.00001, 0.0001, 0.001])
OUTLIER_GRID = jnp.array([0.01, 0.001, 0.0001])
# VARIANCE_GRID = jnp.array([0.001])
# OUTLIER_GRID = jnp.array([ 0.0001])
OUTLIER_VOLUME = 100.0

grid_params = [
    (0.2, jnp.pi, (11,11,11)), (0.1, jnp.pi/3, (11,11,11)), (0.05, 0.0, (11,11,1)),
    (0.05, jnp.pi/5, (11,11,11)), (0.02, 2*jnp.pi, (5,5,51))
]
contact_param_gridding_schedule = [
    b.utils.make_translation_grid_enumeration_3d(
        -x, -x, -ang,
        x, x, ang,
        *nums
    )
    for (x,ang,nums) in grid_params
]
key = jax.random.PRNGKey(500)

importance_jit = jax.jit(b.genjax.model.importance)

contact_enumerators = [b.genjax.make_enumerator([f"contact_params_{i}", "variance", "outlier_prob"]) for i in range(5)]
add_object_jit = jax.jit(b.genjax.add_object)

def c2f_contact_update(trace_, key,  number, contact_param_deltas, VARIANCE_GRID, OUTLIER_GRID):
    contact_param_grid = contact_param_deltas + trace_[f"contact_params_{number}"]
    scores = contact_enumerators[number][3](trace_, key, contact_param_grid, VARIANCE_GRID, OUTLIER_GRID)
    i,j,k = jnp.unravel_index(scores.argmax(), scores.shape)
    return contact_enumerators[number][0](
        trace_, key,
        contact_param_grid[i], VARIANCE_GRID[j], OUTLIER_GRID[k]
    )
c2f_contact_update_jit = jax.jit(c2f_contact_update, static_argnames=("number",))

V_VARIANT = 0
O_VARIANT = 0
HIERARCHICAL_BAYES = True

for scene_id in tqdm(range(200)):
    if HIERARCHICAL_BAYES:
        filename = f"data/inferred_hb_{scene_id}.joblib"
    else:
        filename = f"data/inferred_{V_VARIANT}_{O_VARIANT}_{scene_id}.joblib"

    if os.path.exists(filename):
        continue

    print("GPU Memory: ", b.utils.get_gpu_memory())
    if HIERARCHICAL_BAYES:
        V_GRID = VARIANCE_GRID
        O_GRID = OUTLIER_GRID
    else:
        V_GRID, O_GRID = jnp.array([VARIANCE_GRID[V_VARIANT]]), jnp.array([OUTLIER_GRID[O_VARIANT]])

    gt_trace = importance_jit(key, *joblib.load(f"data/trace_{scene_id}.joblib"))[1][1]
    choices = gt_trace.get_choices()
    key, (_,trace) = importance_jit(key, choices, (jnp.arange(1), jnp.arange(22), *gt_trace.get_args()[2:]))

    for _ in range(3):
        all_paths = []
        for obj_id in tqdm(range(len(b.RENDERER.meshes)-1)):
            path = []
            trace_ = add_object_jit(trace, key, obj_id, 0, 2,3)
            number = b.genjax.get_contact_params(trace_).shape[0] - 1
            path.append(trace_)
            for c2f_iter in range(len(contact_param_gridding_schedule)):
                trace_ = c2f_contact_update_jit(trace_, key, number,
                    contact_param_gridding_schedule[c2f_iter], V_GRID, O_GRID)
                path.append(trace_)
            # for c2f_iter in range(len(contact_param_gridding_schedule)):
            #     trace_ = c2f_contact_update_jit(trace_, key, number,
            #         contact_param_gridding_schedule[c2f_iter], VARIANCE_GRID, OUTLIER_GRID)
            all_paths.append(
                path
            )
        
        scores = jnp.array([t[-1].get_score() for t in all_paths])
        print(scores)
        normalized_scores = b.utils.normalize_log_scores(scores)
        trace = all_paths[jnp.argmax(scores)][-1]
    
    print(b.genjax.get_indices(gt_trace))
    print(b.genjax.get_indices(trace))

    joblib.dump((trace.get_choices(), trace.get_args()), filename)
    del trace
    del gt_trace