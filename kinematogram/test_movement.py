import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as j
import bayes3d as b
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

import genjax
from genjax import GenerativeFunction, ChoiceMap, Selection, trace

intrinsics = b.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=6.0
)

b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/cube.obj"))