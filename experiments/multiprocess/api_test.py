
import pickle as pkl
import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import os
import trimesh
import jax.numpy as jnp

with open("data.pkl","rb") as f:
    d = pkl.load(f)

from api import spatial_elimination
good_poses = spatial_elimination(d)

import matplotlib.pyplot as plt
plt.scatter(good_poses[:,0], good_poses[:, 2])
plt.savefig("floor_plane.png")


