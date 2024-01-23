import jax
import jax.numpy as jnp

import bayes3d as b


def test_estimate_transform_between_clouds():
    key = jax.random.PRNGKey(500)
    c1 = jax.random.uniform(jax.random.PRNGKey(0), (10, 3)) * 5.0
    random_pose = b.distributions.gaussian_vmf_zero_mean(key, 0.1, 1.0)
    c2 = b.t3d.apply_transform(c1, random_pose)

    estimated = b.estimate_transform_between_clouds(c1, c2)
    assert jnp.isclose(b.apply_transform(c1, estimated), c2, atol=1e-5).all()
