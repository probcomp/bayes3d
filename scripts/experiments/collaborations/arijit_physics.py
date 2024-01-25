import genjax
import jax
import jax.numpy as jnp

import bayes3d as b
import bayes3d.genjax

b.setup_visualizer()


@genjax.gen
def body_fun(prev):
    (t, pose, velocity) = prev
    velocity = b.gaussian_vmf_pose(velocity, 0.01, 10000.0) @ "velocity"
    pose = b.gaussian_vmf_pose(pose @ velocity, 0.01, 10000.0) @ "pose"
    # Render
    return (t + 1, pose, velocity)


# Creating a `SwitchCombinator` via the preferred `new` class method.


@genjax.gen
def model(T):
    pose = b.uniform_pose(jnp.ones(3) * -1.0, jnp.ones(3) * 1.0) @ "init_pose"
    velocity = b.gaussian_vmf_pose(jnp.eye(4), 0.01, 10000.0) @ "init_velocity"
    _evolve = (
        genjax.UnfoldCombinator.new(body_fun, 100)(50, (0, pose, velocity)) @ "dynamics"
    )
    return 1.0


key = jax.random.PRNGKey(314159)
tr = model.simulate(key, (10,))
poses = tr["dynamics"]["pose"]
for i in range(poses.shape[0]):
    b.show_pose(f"{i}", poses[i])

# TODO:
# 1. Add rendering and images likelihood
# Do simple SMC tracking of one object
# 2. Make this multiobject
