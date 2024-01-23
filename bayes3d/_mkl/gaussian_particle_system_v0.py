# AUTOGENERATED! DO NOT EDIT! File to edit: ../../scripts/_mkl/notebooks/07a - Gaussian particle system v0.ipynb.

# %% auto 0
__all__ = ['normal_cdf', 'normal_pdf', 'normal_logpdf', 'inv', 'key', 'Array', 'Shape', 'Bool', 'Float', 'Int', 'Pose']

# %% ../../scripts/_mkl/notebooks/07a - Gaussian particle system v0.ipynb 2
import bayes3d as b3d
import trimesh
import os
from bayes3d._mkl.utils import *
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
from functools import partial
import genjax
from bayes3d.camera import Intrinsics, K_from_intrinsics, camera_rays_from_intrinsics
from bayes3d.transforms_3d import transform_from_pos_target_up, add_homogenous_ones, unproject_depth
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax.math import lambertw



normal_cdf    = jax.scipy.stats.norm.cdf
normal_pdf    = jax.scipy.stats.norm.pdf
normal_logpdf = jax.scipy.stats.norm.logpdf
inv = jnp.linalg.inv

key = jax.random.PRNGKey(0)

# %% ../../scripts/_mkl/notebooks/07a - Gaussian particle system v0.ipynb 4
from typing import Any, NamedTuple
import numpy as np
import jax
import jaxlib

Array = np.ndarray | jax.Array
Shape = int | tuple[int, ...]
Bool = Array
Float = Array
Int = Array

Pose = tuple[jnp.ndarray, jnp.ndarray]


class Pose(NamedTuple):
    quat: jnp.ndarray
    position: jnp.ndarray