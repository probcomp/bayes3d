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

@dataclass
class GaussianVMFPose(ExactDensity):
    def sample(self, key, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf_sample(key, pose_mean, var, concentration)

    def logpdf(self, pose, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf_logpdf(pose, pose_mean, var, concentration)

@dataclass
class ImageLikelihood(ExactDensity):
    def sample(self, key, img, variance, outlier_prob, outlier_volume):
        return img

    def logpdf(self, image, s, variance, outlier_prob, outlier_volume):
        return b.threedp3_likelihood(image, s, variance, outlier_prob, outlier_volume, 5)

@dataclass
class ContactParamsUniform(ExactDensity):
    def sample(self, key, low, high):
        return jax.random.uniform(key, shape=(3,)).reshape(-1,1) * (high - low) + low

    def logpdf(self, sampled_val, low, high, **kwargs):
        valid = ((low <= sampled_val) & (sampled_val <= high))
        log_probs = jnp.log((valid * 1.0) * (jnp.ones_like(sampled_val) / (high-low)))
        return log_probs.sum()

@dataclass
class UniformDiscreteArray(ExactDensity):
    def sample(self, key, vals, arr):
        return jax.random.choice(key, vals, shape=arr.shape)

    def logpdf(self, sampled_val, vals, arr,**kwargs):
        return jnp.log(1.0 / (vals.shape[0])) * arr.shape[0]

@dataclass
class UniformDiscrete(ExactDensity):
    def sample(self, key, vals):
        return jax.random.choice(key, vals)

    def logpdf(self, sampled_val, vals,**kwargs):
        return jnp.log(1.0 / (vals.shape[0]))


gaussian_vmf_pose = GaussianVMFPose()
image_likelihood = ImageLikelihood()
contact_params_uniform = ContactParamsUniform()
uniform_discrete = UniformDiscrete()
uniform_discrete_array = UniformDiscreteArray()

def viz_trace_meshcat(trace, colors=None):
    b.clear()
    key = jax.random.PRNGKey(10)
    b.show_cloud("1", trace["image"].reshape(-1,3))
    b.show_cloud("2", trace.get_retval()[0].reshape(-1,3),color=b.RED)
    # noisy_point_cloud_image = jax.random.multivariate_normal(
    #     key, trace.observation[:,:,:3].reshape(-1,3), jnp.eye(3)*trace.variance
    # )
    # b.show_cloud("2", noisy_point_cloud_image.reshape(-1,3), color=b.RED)
    indices = trace.get_retval()[1]
    print(indices)
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(indices)))
    for i in range(len(indices)):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[indices[i]],color=colors[i])
        b.set_pose(f"obj_{i}", trace.get_retval()[2][i])

@genjax.gen
def tabletop_model(array, possible_object_indices, root_poses, all_box_dims, outlier_volume):
    indices = jnp.array([-1])
    for i in range(array.shape[0]):
        index = uniform_discrete(possible_object_indices) @ f"id_{i}"
        indices = jnp.concatenate([indices, jnp.array([index])])
    indices = indices[1:]
        
    contact_params = contact_params_uniform(jnp.array([[-0.2,-0.2, -2*jnp.pi]]), jnp.array([[0.2,0.2, 2*jnp.pi]]), array) @ "contact_params"
    parents = uniform_discrete_array(jnp.arange(-1,1), array) @ "parents"
    face_parent = uniform_discrete_array(jnp.arange(2,3), array) @ "face_parent"
    face_child = uniform_discrete_array(jnp.arange(3,4), array) @ "face_child"
    box_dims = all_box_dims[indices]
    poses = b.scene_graph.poses_from_scene_graph(root_poses, box_dims, parents, contact_params, face_parent, face_child)
    rendered = b.RENDERER.render_jax(
        poses , indices
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(0.00001, 0.1) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(0.00001, 0.01) @ "outlier_prob"
    image = image_likelihood(rendered, variance, outlier_prob, outlier_volume) @ "image"
    return rendered, indices, poses, image

get_rendered_image = lambda trace: trace.get_retval()[0]
get_indices = lambda trace: trace.get_retval()[1]
get_poses = lambda trace: trace.get_retval()[2]

simulate_jit = jax.jit(tabletop_model.simulate)
update_jit = jax.jit(tabletop_model.update)
importance_jit = jax.jit(tabletop_model.importance)
importance_parallel_jit = jax.jit(jax.vmap(tabletop_model.importance, in_axes=(0, None, None)))

enumerator_trace = lambda trace, key, c, v, o : trace.update(
    key,
    genjax.choice_map({
        "contact_params": c, 
        "variance": v,
        "outlier_prob": o,
    }),
    jtu.tree_map(lambda v: Diff(v, NoChange), trace.args),
)[1][2]
enumerator_trace_jit = jax.jit(enumerator_trace)
enumerator_trace_vmap = jax.vmap(jax.vmap(jax.vmap(enumerator_trace, in_axes=(None, None, None, None, 0)), in_axes=(None, None, None, 0, None)), in_axes=(None, None, 0, None, None))
enumerator_trace_vmap_jit = jax.jit(enumerator_trace_vmap)

enumerator_score = lambda trace, key, c, v, o: enumerator_trace(trace, key, c, v, o).get_score()
enumerator_score_jit = jax.jit(enumerator_score)
enumerator_score_vmap = jax.vmap(jax.vmap(jax.vmap(enumerator_score, in_axes=(None, None, None, None, 0)), in_axes=(None, None, None, 0, None)), in_axes=(None, None, 0, None, None))
enumerator_score_vmap_jit = jax.jit(enumerator_score_vmap)