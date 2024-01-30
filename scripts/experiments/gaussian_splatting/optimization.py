import os

import bayes3d as b
import jax.numpy as jnp
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
mesh_path = os.path.join(model_dir, "obj_" + "{}".format(3).rjust(6, "0") + ".ply")
mesh = b.utils.load_mesh(mesh_path)
vertices = torch.tensor(np.array(jnp.array(mesh.vertices) / 1000.0), device=device)
