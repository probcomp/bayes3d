import jax.numpy as jnp
import jax
from jax3dp3.transforms_3d import transform_from_axis_angle

def angle_axis_helper_edgecase(newZ):
    zUnit = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.array([0.0, 1.0, 0.0])
    geodesicAngle = jax.lax.cond(jnp.allclose(newZ, zUnit, atol=1e-3), lambda:0.0, lambda:jnp.pi)
    return axis, geodesicAngle


def angle_axis_helper(newZ): 
    zUnit = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.cross(zUnit, newZ)
    theta = jax.lax.asin(jax.lax.clamp(-1.0, jnp.linalg.norm(axis), 1.0))
    geodesicAngle = jax.lax.cond(jnp.dot(zUnit, newZ) > 0, lambda:theta, lambda:jnp.pi - theta)  
    return axis, geodesicAngle


def geodesicHopf_select_axis(newZ, planarAngle):
    # newZ should be a normalized vector
    # returns a 4x4 quaternion
    
    zUnit = jnp.array([1.0, 0.0, 0.0])

    # todo: implement cases where newZ is approx. -zUnit or approx. zUnit
    axis, geodesicAngle = jax.lax.cond(jnp.allclose(jnp.abs(newZ), jnp.array([1,0,0]), atol=1e-3), angle_axis_helper_edgecase, angle_axis_helper, newZ)

    return (transform_from_axis_angle(axis, geodesicAngle) @ transform_from_axis_angle(zUnit, planarAngle))


def fibonacci_sphere(samples):
    phi = jnp.pi * (3 - jnp.sqrt(5))  # golden angle
    def fib_point(i):
        y = 1 - (i / (samples - 1)) * 2
        radius = jnp.sqrt(1 - y * y)
        theta = phi * i
        x = jnp.cos(theta) * radius
        z = jnp.sin(theta) * radius
        return jnp.array([x, y, z])
        
    fib_sphere = jax.vmap(fib_point, in_axes=(0))
    points = jnp.arange(samples)
    return fib_sphere(points)


def get_rotation_proposals(sample, rot_sample):
    unit_sphere_directions = fibonacci_sphere(sample)
    geodesicHopf_select_axis_vmap = jax.vmap(jax.vmap(geodesicHopf_select_axis, in_axes=(0,None)), in_axes=(None,0))
    stepsize = 2*jnp.pi / rot_sample
    rotation_proposals = geodesicHopf_select_axis_vmap(unit_sphere_directions, jnp.arange(0, 2*jnp.pi, stepsize)).reshape(-1, 4, 4)
    return rotation_proposals
