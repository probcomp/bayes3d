from dataclasses import dataclass
import jax.numpy as jnp
import bayes3d as b
import jax

@dataclass
class Trace:
    poses: jnp.ndarray
    ids: list
    variance: float
    outlier_prob: float
    outlier_volume: float
    observation: jnp.ndarray

    def __str__(self):
        return f"variance: {self.variance} outlier_prob: {self.outlier_prob} outlier_volume: {self.outlier_volume}\n ids: {self.ids} poses: {self.poses}"


def render_image(trace, renderer):
    reconstruction = renderer.render_multiobject(
        trace.poses , trace.ids
    )
    return reconstruction

def score_trace(trace, renderer, filter_size=3, mask=None):
    reconstruction = render_image(trace, renderer)
    if mask is None:
        p = b.threedp3_likelihood_jit(
            trace.observation, reconstruction[:,:,:3],
            trace.variance, trace.outlier_prob, trace.outlier_volume,
            filter_size
        )
        return p
    else:
        p = b.threedp3_likelihood_per_pixel_jit(
            trace.observation, reconstruction[:,:,:3],
            trace.variance, trace.outlier_prob, trace.outlier_volume,
            filter_size
        )
        return (p*mask).sum()

def viz_trace_meshcat(trace, renderer):
    b.clear()
    key = jax.random.PRNGKey(10)
    b.show_cloud("1", trace.observation.reshape(-1,3))
    # noisy_point_cloud_image = jax.random.multivariate_normal(
    #     key, trace.observation[:,:,:3].reshape(-1,3), jnp.eye(3)*trace.variance
    # )
    # b.show_cloud("2", noisy_point_cloud_image.reshape(-1,3), color=b.RED)

    colors = b.viz.distinct_colors(10)
    for i in range(len(trace.ids)):
        b.show_trimesh(f"obj_{i}", renderer.meshes[trace.ids[i]],color=colors[i])
        b.set_pose(f"obj_{i}", trace.poses[i])



@dataclass
class Traces:
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

def render_images(traces, renderer):
    reconstruction = renderer.render_multiobject_parallel(
        traces.all_poses , traces.ids
    )
    return reconstruction


def score_traces(traces, renderer, filter_size=3):
    reconstruction = render_images(traces, renderer)
    p = b.threedp3_likelihood_full_hierarchical_bayes_jit(
        traces.observation, reconstruction[:,:,:,:3],
        traces.all_variances, traces.all_outlier_prob, traces.outlier_volume,
        filter_size
    )
    return p