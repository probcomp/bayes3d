# AUTOGENERATED! DO NOT EDIT! File to edit: ../../scripts/_mkl/notebooks/30 - Table Scene Model.ipynb.

# %% auto 0
__all__ = ['normal_logpdf', 'normal_pdf', 'truncnorm_logpdf', 'truncnorm_pdf', 'inv', 'logaddexp', 'logsumexp', 'key',
           'make_table_scene_model']

# %% ../../scripts/_mkl/notebooks/30 - Table Scene Model.ipynb 2
import bayes3d as b3d
import bayes3d.genjax
import joblib
from tqdm import tqdm
import os
import jax.numpy as jnp
import jax
from jax import jit, vmap
import numpy as np
import genjax
import trimesh
import matplotlib.pyplot as plt
from bayes3d.genjax.genjax_distributions import *

# console = genjax.pretty(show_locals=False)

# %% ../../scripts/_mkl/notebooks/30 - Table Scene Model.ipynb 3
from jax.scipy.spatial.transform import Rotation
from scipy.stats import truncnorm as scipy_truncnormal

normal_logpdf    = jax.scipy.stats.norm.logpdf
normal_pdf       = jax.scipy.stats.norm.pdf
truncnorm_logpdf = jax.scipy.stats.truncnorm.logpdf
truncnorm_pdf    = jax.scipy.stats.truncnorm.pdf

inv       = jnp.linalg.inv
logaddexp = jnp.logaddexp
logsumexp = jax.scipy.special.logsumexp

key = jax.random.PRNGKey(0)

# %% ../../scripts/_mkl/notebooks/30 - Table Scene Model.ipynb 4
from bayes3d._mkl.utils import keysplit
from bayes3d._mkl.plotting import *

# %% ../../scripts/_mkl/notebooks/30 - Table Scene Model.ipynb 11
def make_table_scene_model():
    """
    Example:

    ```
    key = keysplit(key)

    model = make_table_scene_model()

    table = jnp.eye(4)
    cam   = b3d.transform_from_pos_target_up(
                jnp.array([0.0, -.5, -.75]), 
                jnp.zeros(3), 
                jnp.array([0.0,-1.0,0.0]))

    args = (
        jnp.arange(3), 
        jnp.arange(22), 
        jnp.array([-jnp.ones(3)*100.0, jnp.ones(3)*100.0]),
        jnp.array([jnp.array([-0.2, -0.2, -2*jnp.pi]), jnp.array([0.2, 0.2, 2*jnp.pi])]),
        b3d.RENDERER.model_box_dims
    )

    ch = genjax.choice_map({
        "parent_0": -1,
        "parent_1":  0,
        "parent_2":  0,
        "camera_pose": cam,
        "root_pose_0": table,
        "id_0": jnp.int32(21), # Atomic Table
        "id_1": jnp.int32(13), # Mug
        "id_2": jnp.int32(2),  # Box
        "face_parent_1": 1,  # That's the top face of the table
        "face_parent_2": 1,  # ...
        "face_child_1": 3,   # That's a bottom face of the mug
        "face_child_2": 3,
    })

    w, tr = model.importance(key, ch , args)
    cam, ps, inds = tr.retval
    X = render(cam, ps, inds)

    # =====================
    plt.imshow(X[...,2])
    ```
    """

    @genjax.gen
    def model(nums, 
              possible_object_indices, 
              pose_bounds, 
              contact_bounds, 
              all_box_dims):
        
        num_objects = len(nums) # this is a hack, otherwise genajx is complaining

        indices        = jnp.array([], dtype=jnp.int32)
        root_poses     = jnp.zeros((0,4,4))
        contact_params = jnp.zeros((0,3))
        faces_parents  = jnp.array([], dtype=jnp.int32)
        faces_child    = jnp.array([], dtype=jnp.int32)
        parents        = jnp.array([], dtype=jnp.int32)

        for i in range(num_objects):

            index  = uniform_discrete(possible_object_indices)    @ f"id_{i}"
            pose   = uniform_pose(pose_bounds[0], pose_bounds[1]) @ f"root_pose_{i}"
            params = contact_params_uniform(contact_bounds[0], contact_bounds[1]) @ f"contact_params_{i}"

            parent_obj  = uniform_discrete(jnp.arange(-1, num_objects - 1)) @ f"parent_{i}"
            parent_face = uniform_discrete(jnp.arange(0,6)) @ f"face_parent_{i}"
            child_face  = uniform_discrete(jnp.arange(0,6)) @ f"face_child_{i}"

            indices        = jnp.concatenate([indices, jnp.array([index])])
            root_poses     = jnp.concatenate([root_poses, pose.reshape(1,4,4)])
            contact_params = jnp.concatenate([contact_params, params.reshape(1,-1)])
            parents        = jnp.concatenate([parents, jnp.array([parent_obj])])
            faces_parents  = jnp.concatenate([faces_parents, jnp.array([parent_face])])
            faces_child    = jnp.concatenate([faces_child, jnp.array([child_face])])
        

        scene = (root_poses, all_box_dims[indices], parents, contact_params, faces_parents, faces_child)
        poses = b.scene_graph.poses_from_scene_graph(*scene)

        camera_pose = uniform_pose(pose_bounds[0], pose_bounds[1]) @ f"camera_pose"

        return camera_pose, poses, indices

    return model
