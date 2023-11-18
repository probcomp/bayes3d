import jax
from dataclasses import dataclass
from genjax.generative_functions.distributions import ExactDensity
import jax.numpy as jnp
import bayes3d as b

@dataclass
class GaussianVMFPose(ExactDensity):
    def sample(self, key, pose_mean, var, concentration, **kwargs):
        return b.distributions.gaussian_vmf(key, pose_mean, var, concentration)

    def logpdf(self, pose, pose_mean, var, concentration, **kwargs):

        return b.distributions.gaussian_vmf_logpdf(pose, pose_mean, var, concentration)

@dataclass
class UniformPose(ExactDensity):
    def sample(self, key, low, high, **kwargs):
        position = jax.random.uniform(key, shape=(3,)) * (high - low) + low
        orientation = b.quaternion_to_rotation_matrix(jax.random.normal(key, shape=(4,)))
        return b.transform_from_rot_and_pos(orientation, position)

    def logpdf(self, pose, low, high, **kwargs):
        position = pose[:3,3]
        valid = ((low <= position) & (position <= high))
        position_score = jnp.log((valid * 1.0) * (jnp.ones_like(position) / (high-low)))
        return position_score.sum() + jnp.pi**2

@dataclass
class ImageLikelihood(ExactDensity):
    def sample(self, key, img, variance, outlier_prob):
        return img

    def logpdf(self, observed_image, latent_image, variance, outlier_prob):
        return b.threedp3_likelihood(
            observed_image, latent_image, variance, outlier_prob,
        )
    
@dataclass
class OldImageLikelihood(ExactDensity):
    def sample(self, key, img, variance, outlier_prob, outlier_volume, focal_length):
        return img

    def logpdf(self, image, s, variance, outlier_prob, outlier_volume, focal_length):
        return b.threedp3_likelihood_old(
            image, s, variance, outlier_prob,
            outlier_volume, focal_length, 3
        )

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
old_image_likelihood = OldImageLikelihood()
contact_params_uniform = ContactParamsUniform()
uniform_discrete = UniformDiscrete()
uniform_discrete_array = UniformDiscreteArray()
uniform_pose = UniformPose()
