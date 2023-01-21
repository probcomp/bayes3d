
import jax3dp3
import jax.numpy as jnp
import numpy as np
import trimesh

def setup(
    h, w, fx, fy, cx, cy, near, far, scaling_factor,
    model_paths, model_names, model_scaling_factor,
):
    jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

    model_box_dims = []
    for path in model_paths:
        mesh = trimesh.load(path)  # 000001 to 000021
        mesh.vertices = mesh.vertices * model_scaling_factor
        model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
        jax3dp3.load_model(mesh)
    model_box_dims = jnp.array(model_box_dims)
    
    return model_box_dims, (h, w, fx, fy, cx, cy, near, far)
