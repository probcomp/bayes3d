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
        return jax.random.uniform(key, shape=(3,)) * (high - low) + low

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
    indices = trace.get_retval()[1]
    if colors is None:
        colors = b.viz.distinct_colors(max(10, len(indices)))
    for i in range(len(indices)):
        b.show_trimesh(f"obj_{i}", b.RENDERER.meshes[indices[i]],color=colors[i])
        b.set_pose(f"obj_{i}", trace.get_retval()[2][i])

@genjax.gen
def tabletop_model(array, possible_object_indices, root_poses, all_box_dims, outlier_volume):

    indices = jnp.array([], dtype=jnp.int32)
    contact_params = jnp.zeros((0,3))
    faces_parents = jnp.array([], dtype=jnp.int32)
    faces_child = jnp.array([], dtype=jnp.int32)
    parents = jnp.array([], dtype=jnp.int32)
    for i in range(array.shape[0]):
        index = b.tabletop_model.uniform_discrete(possible_object_indices) @ f"id_{i}"
        indices = jnp.concatenate([indices, jnp.array([index])])
        
        params = contact_params_uniform(
            jnp.array([-0.2,-0.2, -2*jnp.pi]), 
            jnp.array([0.2,0.2, 2*jnp.pi])) @ f"contact_params_{i}"
        contact_params = jnp.concatenate([contact_params, params.reshape(1,-1)])

        parent_obj = b.tabletop_model.uniform_discrete(jnp.arange(-1,array.shape[0] - 1)) @ f"parent_{i}"
        parents = jnp.concatenate([parents, jnp.array([parent_obj])])
        parent_face = b.tabletop_model.uniform_discrete(jnp.arange(0,6)) @ f"face_parent_{i}"
        faces_parents = jnp.concatenate([faces_parents, jnp.array([parent_face])])
        child_face = b.tabletop_model.uniform_discrete(jnp.arange(0,6)) @ f"face_child_{i}"
        faces_child = jnp.concatenate([faces_child, jnp.array([child_face])])
    
    box_dims = all_box_dims[indices]
    poses = b.scene_graph.poses_from_scene_graph(
        root_poses, box_dims, parents, contact_params, faces_parents, faces_child)
    rendered = b.RENDERER.render_jax(
        poses , indices
    )[...,:3]

    variance = genjax.distributions.tfp_uniform(0.00001, 0.1) @ "variance"
    outlier_prob  = genjax.distributions.tfp_uniform(0.00001, 0.01) @ "outlier_prob"
    image = b.tabletop_model.image_likelihood(rendered, variance, outlier_prob, outlier_volume) @ "image"
    return rendered, indices, poses, parents, contact_params, faces_parents, faces_child

get_rendered_image = lambda trace: trace.get_retval()[0]
get_indices = lambda trace: trace.get_retval()[1]
get_poses = lambda trace: trace.get_retval()[2]


enumerator = lambda trace, key, address, c: trace.update(
    key,
    genjax.choice_map({
        address: c, 
    }),
    jtu.tree_map(lambda v: Diff(v, UnknownChange), trace.args),
)[1][2]
enumerator_vmap_jit = jax.jit(jax.vmap(
    enumerator, in_axes=(None, None, None, 0)), static_argnames=("address",))
enumerator_vmap_score_jit = jax.jit(jax.vmap(
    lambda trace, key, address, c: enumerator(trace, key, address, c).get_score(), in_axes=(None, None, None, 0)), static_argnames=("address",))