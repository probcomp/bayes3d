import jax.numpy as jnp
import jax
from jax3dp3.transforms_3d import transform_from_axis_angle, transform_from_pos

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


def make_rotation_grid_enumeration(fibonacci_sphere_points, num_planar_angle_points):
    unit_sphere_directions = fibonacci_sphere(fibonacci_sphere_points)
    geodesicHopf_select_axis_vmap = jax.vmap(jax.vmap(geodesicHopf_select_axis, in_axes=(0,None)), in_axes=(None,0))
    stepsize = 2*jnp.pi / num_planar_angle_points
    rotation_proposals = geodesicHopf_select_axis_vmap(unit_sphere_directions, jnp.arange(0, 2*jnp.pi, stepsize)).reshape(-1, 4, 4)
    return rotation_proposals

def make_translation_grid_enumeration(min_x,min_y,min_z, max_x,max_y,max_z, num_x,num_y,num_z):
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(min_x, max_x, num_x),
        jnp.linspace(min_y, max_y, num_y),
        jnp.linspace(min_z, max_z, num_z)
    ),
        axis=-1)
    deltas = deltas.reshape(-1,3)
    return jax.vmap(transform_from_pos)(deltas)

def make_grid_enumeration(min_x,min_y,min_z, max_x,max_y,max_z, num_x,num_y,num_z, fibonacci_sphere_points, num_planar_angle_points):
    rotations = make_rotation_grid_enumeration(fibonacci_sphere_points, num_planar_angle_points)
    translations = make_translation_grid_enumeration(min_x,min_y,min_z, max_x,max_y,max_z, num_x,num_y,num_z)
    all_proposals = jnp.einsum("aij,bjk->abik", rotations, translations).reshape(-1, 4, 4)
    return all_proposals



