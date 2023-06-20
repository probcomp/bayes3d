from typing import NamedTuple, Any
import jax.numpy as jnp
import bayes3d as b
import jax


class Trace(NamedTuple):
    poses: jnp.ndarray
    ids: jnp.ndarray
    variance: float
    outlier_prob: float
    outlier_volume: float
    observation: jnp.ndarray

    def __str__(self):
        return f"variance: {self.variance} outlier_prob: {self.outlier_prob} outlier_volume: {self.outlier_volume}\n ids: {self.ids} poses: {self.poses}"


def render_image(trace):
    reconstruction = b.RENDERER.render_multiobject(
        trace.poses , trace.ids
    )
    return reconstruction
render_image_jit = jax.jit(render_image)

def score_trace(trace, filter_size=3, mask=None):
    reconstruction = render_image(trace)
    p = b.threedp3_likelihood_jit(
        trace.observation, reconstruction[:,:,:3],
        trace.variance, trace.outlier_prob, trace.outlier_volume,
        filter_size
    )
    return p
score_trace_jit = jax.jit(score_trace)

def viz_trace_meshcat(trace, colors=None):
    b.clear()
    key = jax.random.PRNGKey(10)
    b.show_cloud("1", trace.observation.reshape(-1,3))
    # noisy_point_cloud_image = jax.random.multivariate_normal(
    #     key, trace.observation[:,:,:3].reshape(-1,3), jnp.eye(3)*trace.variance
    # )
    # b.show_cloud("2", noisy_point_cloud_image.reshape(-1,3), color=b.RED)
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(trace.ids)))
    for i in range(len(trace.ids)):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[trace.ids[i]],color=colors[i])
        b.set_pose(f"obj_{i}", trace.poses[i])



class Traces(NamedTuple):
    all_poses: jnp.ndarray
    ids: list
    all_variances: float
    all_outlier_prob: float
    outlier_volume: float
    observation: jnp.ndarray

    def __getitem__(self, key):
        assert len(key) == 3
        return Trace(
            self.all_poses[:, key[0]],
            self.ids,
            self.all_variances[key[1]],
            self.all_outlier_prob[key[2]],
            self.outlier_volume,
            self.observation
        )

def render_images(traces):
    reconstruction = b.RENDERER.render_multiobject_parallel(
        traces.all_poses , traces.ids
    )
    return reconstruction
render_images_jit = jax.jit(render_images)


def score_traces(traces, filter_size=3):
    reconstruction = render_images(traces)
    p = b.threedp3_likelihood_full_hierarchical_bayes_jit(
        traces.observation, reconstruction[:,:,:,:3],
        traces.all_variances, traces.all_outlier_prob, traces.outlier_volume,
        filter_size
    )
    return p
score_traces_jit = jax.jit(score_traces)


############################



# class TraceSceneGraph(NamedTuple):
#     root_poses: jnp.ndarray
#     box_dims: jnp.ndarray
#     parents: jnp.ndarray
#     contact_params: jnp.ndarray
#     face_parent: jnp.ndarray
#     face_child: jnp.ndarray
#     ids: jnp.ndarray
#     variance: float
#     outlier_prob: float
#     outlier_volume: float
#     observation: jnp.ndarray

# def render_image(trace):
#     poses = b.scene_graph.poses_from_scene_graph(trace.root_poses, trace.box_dims, trace.parents, trace.contact_params, trace.face_parent, trace.face_child)
#     reconstruction = b.RENDERER.render_multiobject(
#         poses , trace.ids
#     )
#     return reconstruction

# def score_trace(trace, filter_size=3):
#     reconstruction = render_image(trace)
#     p = b.threedp3_likelihood(
#         trace.observation, reconstruction[:,:,:3],
#         trace.variance, trace.outlier_prob, trace.outlier_volume,
#         filter_size
#     )
#     return p
# score_trace_jit = jax.jit(score_trace)

# def add_object_to_trace(trace, pose, box_dim, parent, contact_param, face_parent, face_child, id):
#     return TraceSceneGraph(
#         jnp.concatenate([trace.root_poses, pose[None,...]]),
#         jnp.concatenate([trace.box_dims, box_dim.reshape(1,3)]),
#         jnp.concatenate([trace.parents, jnp.array([parent])]),
#         jnp.concatenate([trace.contact_params, contact_param.reshape(1,3)]),
#         jnp.concatenate([trace.face_parent, jnp.array([face_parent])]),
#         jnp.concatenate([trace.face_child, jnp.array([face_child])]),
#         jnp.concatenate([trace.ids, jnp.array([id])]),
#         trace.variance,
#         trace.outlier_prob,
#         trace.outlier_volume,
#         trace.observation
#     )


# contact_poses_parallel = jax.vmap(
#     b.scene_graph.poses_from_scene_graph,
#     in_axes=(None, None, None, 0, None, None), out_axes=1
# )

# class TracesSceneGraph(NamedTuple):
#     root_poses: jnp.ndarray
#     box_dims: jnp.ndarray
#     parents: jnp.ndarray
#     all_contact_params: jnp.ndarray
#     face_parent: jnp.ndarray
#     face_child: jnp.ndarray
#     ids: jnp.ndarray
#     all_variances: float
#     all_outlier_prob: float
#     outlier_volume: float
#     observation: jnp.ndarray
    
#     def __getitem__(self, key):
#         assert len(key) == 3
#         return TraceSceneGraph(
#             self.root_poses,
#             self.box_dims,
#             self.parents,
#             self.all_contact_params[key[0]],
#             self.face_parent,
#             self.face_child,
#             self.ids,
#             self.all_variances[key[1]],
#             self.all_outlier_prob[key[2]],
#             self.outlier_volume,
#             self.observation
#         )

# def render_images(traces):
#     poses = contact_poses_parallel(traces.root_poses, traces.box_dims, traces.parents, traces.all_contact_params, traces.face_parent, traces.face_child)
#     reconstruction = b.RENDERER.render_multiobject_parallel(
#         poses , traces.ids
#     )
#     return reconstruction

# def score_traces(traces, filter_size=3):
#     reconstruction = render_images(traces)
#     p = b.threedp3_likelihood_full_hierarchical_bayes_jit(
#         traces.observation, reconstruction[:,:,:,:3],
#         traces.all_variances, traces.all_outlier_prob, traces.outlier_volume,
#         filter_size
#     )
#     return p
# score_traces_jit = jax.jit(score_traces)