from genjax.generative_functions.distributions import ExactDensity
import bayes3d as b
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import genjax
import jax
import os
import jax.tree_util as jtu
from tqdm import tqdm
from genjax._src.core.transforms.incremental import NoChange
from genjax._src.core.transforms.incremental import UnknownChange
from genjax._src.core.transforms.incremental import Diff
from collections import namedtuple
import inspect
from .genjax_distributions import *

@genjax.gen
def model(array, possible_object_indices, pose_bounds, contact_bounds, all_box_dims):
    indices = jnp.array([], dtype=jnp.int32)
    root_poses = jnp.zeros((0,4,4))
    contact_params = jnp.zeros((0,3))
    faces_parents = jnp.array([], dtype=jnp.int32)
    faces_child = jnp.array([], dtype=jnp.int32)
    parents = jnp.array([], dtype=jnp.int32)
    for i in range(array.shape[0]):
        parent_obj = uniform_discrete(jnp.arange(-1,array.shape[0] - 1)) @ f"parent_{i}"
        parent_face = uniform_discrete(jnp.arange(0,6)) @ f"face_parent_{i}"
        child_face = uniform_discrete(jnp.arange(0,6)) @ f"face_child_{i}"
        index = uniform_discrete(possible_object_indices) @ f"id_{i}"

        pose = uniform_pose(
            pose_bounds[0], 
            pose_bounds[1], 
        ) @ f"root_pose_{i}"

        params = contact_params_uniform(
            contact_bounds[0], 
            contact_bounds[1]
        ) @ f"contact_params_{i}"


        indices = jnp.concatenate([indices, jnp.array([index])])
        root_poses = jnp.concatenate([root_poses, pose.reshape(1,4,4)])
        contact_params = jnp.concatenate([contact_params, params.reshape(1,-1)])
        parents = jnp.concatenate([parents, jnp.array([parent_obj])])
        faces_parents = jnp.concatenate([faces_parents, jnp.array([parent_face])])
        faces_child = jnp.concatenate([faces_child, jnp.array([child_face])])
    
    box_dims = all_box_dims[indices]
    poses = b.scene_graph.poses_from_scene_graph(
        root_poses, box_dims, parents, contact_params, faces_parents, faces_child)

    camera_pose = uniform_pose(
        pose_bounds[0], 
        pose_bounds[1], 
    ) @ f"camera_pose"

    rendered = b.RENDERER.render(
        jnp.linalg.inv(camera_pose) @ poses , indices
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(0.00000000001, 10000.0) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(-0.01, 10000.0) @ "outlier_prob"
    image = image_likelihood(rendered, variance, outlier_prob) @ "image"
    return rendered, indices, poses, parents, contact_params, faces_parents, faces_child, root_poses

get_rendered_image = lambda trace: trace.get_retval()[0]
get_indices = lambda trace: trace.get_retval()[1]
get_poses = lambda trace: trace.get_retval()[2]
get_parents = lambda trace: trace.get_retval()[3]
get_contact_params = lambda trace: trace.get_retval()[4]
get_faces_parents = lambda trace: trace.get_retval()[5]
get_faces_child = lambda trace: trace.get_retval()[6]
get_root_poses = lambda trace: trace.get_retval()[7]

get_outlier_volume = lambda trace: trace.get_args()[5]
get_focal_length = lambda trace: trace.get_args()[6]
get_far_plane = lambda trace: trace.get_args()[7]

def add_object(trace, key, obj_id, parent, face_parent, face_child):
    N = b.get_indices(trace).shape[0] + 1
    choices = trace.get_choices()
    choices[f"parent_{N-1}"] = parent
    choices[f"id_{N-1}"] = obj_id
    choices[f"face_parent_{N-1}"] = face_parent
    choices[f"face_child_{N-1}"] = face_child
    choices[f"contact_params_{N-1}"] = jnp.zeros(3)
    return model.importance(key, choices,
        (jnp.arange(N), *trace.get_args()[1:])
    )[1]

add_object_jit = jax.jit(add_object)

def print_trace(trace):
    print("""
    SCORE: {:0.7f}
    VARIANCE: {:0.7f}
    OUTLIER_PROB {:0.7f}
    """.format(trace.get_score(), trace["variance"], trace["outlier_prob"]))

def viz_trace_meshcat(trace, colors=None):
    b.clear()
    b.show_cloud("1", b.apply_transform_jit(trace["image"].reshape(-1,3), trace["camera_pose"]))
    b.show_cloud("2", b.apply_transform_jit(get_rendered_image(trace).reshape(-1,3), trace["camera_pose"]),color=b.RED)
    indices = trace.get_retval()[1]
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(indices)))
    for i in range(len(indices)):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[indices[i]],color=colors[i])
        b.set_pose(f"obj_{i}", trace.get_retval()[2][i])
    b.show_pose(f"camera_pose", trace["camera_pose"])



def make_onehot(n, i, hot=1, cold=0):
    return tuple(cold if j != i else hot for j in range(n))

def multivmap(f, args=None):
    if args is None:
        args = (True,) * len(inspect.signature(f).parameters)
    multivmapped = f
    for (i, ismapped) in reversed(list(enumerate(args))):
        if ismapped:
            multivmapped = jax.vmap(multivmapped, in_axes=make_onehot(len(args), i, hot=0, cold=None))
    return multivmapped

Enumerator = namedtuple("Enumerator",["enum_f", "score", "enum_f_vmap", "score_vmap"])


def make_enumerator(addresses):
    def enumerator(trace, key, *args):
        return trace.update(
            key,
            genjax.choice_map({
                addr: c for (addr, c) in zip(addresses, args)
            }),
            tuple(map(lambda v: Diff(v, UnknownChange), trace.args)),
        )[2]
    
    def enumerator_score(trace, key, *args):
        return enumerator(trace, key, *args).get_score()
    # return Enumerator(jax.jit(enumerator), jax.jit(enumerator_score), jax.jit(multivmap(enumerator, (False, False,) + (True,) * len(addresses))), jax.jit(multivmap(enumerator_score, (False, False,) + (True,) * len(addresses))))
    return jax.jit(enumerator), jax.jit(enumerator_score), jax.jit(multivmap(enumerator, (False, False,) + (True,) * len(addresses))), jax.jit(multivmap(enumerator_score, (False, False,) + (True,) * len(addresses)))

def make_unknown_change_argdiffs(trace):
    return tuple(map(lambda v: Diff(v, UnknownChange), trace.args))

def viz_trace_rendered_observed(trace, scale = 2):
    return b.viz.hstack_images(
        [
            b.viz.scale_image(b.get_depth_image(get_rendered_image(trace)[...,2]), scale),
            b.viz.scale_image(b.get_depth_image(trace["image"][...,2]), scale)
        ]
    )

def get_pixelwise_scores(trace, filter_size):
    log_scores_per_pixel = b.threedp3_likelihood_per_pixel_jit(
        trace["image"],
        b.get_rendered_image(trace),
        trace["variance"],
        trace["outlier_prob"],
        get_outlier_volume(trace),
        get_focal_length(trace),
        filter_size
    )
    return log_scores_per_pixel

def update_address(trace, key, address, value):
    return trace.update(
        key,
        genjax.choice_map({
            address: value
        }),
        tuple(map(lambda v: Diff(v, UnknownChange), trace.args)),
    )[2]

