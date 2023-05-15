import bayes3d as b
import numpy as np
import jax.numpy as jnp
import jax
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=200.0, fy=200.0,
    cx=50.0, cy=150.0,
    near=0.001, far=6.0
)

renderer = b.Renderer(intrinsics)
renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))
renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/sphere.obj"))

poses = jnp.tile(jnp.eye(4)[None,None,...], (2,200,1,1))
gt_image = renderer.render_multiobject_parallel(poses, [0,1])[0,:,:,:3]

def score_traces_(ids, poses, observation, all_variances, all_outlier_prob, outlier_volume):
    def f_(poses, observation, all_variances, all_outlier_prob, outlier_volume):
        reconstruction = renderer.render_multiobject_parallel(
            poses, ids
        )
        score = (jax.vmap(jax.vmap(jax.vmap(
            b.threedp3_likelihood,
        in_axes=(None, None, None, 0, None, None)),
        in_axes=(None, None, 0, None, None, None)),
        in_axes=(None, 0, None, None, None, None)
    ))(
            observation, reconstruction[:,:,:,:3],
            all_variances, all_outlier_prob, outlier_volume,
            3
    )
        return score
    return jax.jit(f_)(poses, observation, all_variances, all_outlier_prob, outlier_volume)

x = score_traces_([0,1],poses, gt_image, jnp.array([0.1, 0.1]), jnp.array([0.1, 0.2, 0.3]), 0.1)
