import jax.numpy as jnp
import jax
from jax3dp3.transforms_3d import transform_from_axis_angle, transform_from_pos

def angle_axis_helper_edgecase(newZ):
    zUnit = jnp.array([0.0, 0.0, 1.0])
    axis = jnp.array([0.0, 1.0, 0.0])
    geodesicAngle = jax.lax.cond(jnp.allclose(newZ, zUnit, atol=1e-3), lambda:0.0, lambda:jnp.pi)
    return axis, geodesicAngle


def angle_axis_helper(newZ): 
    zUnit = jnp.array([0.0, 0.0, 1.0])
    axis = jnp.cross(zUnit, newZ)
    theta = jax.lax.asin(jax.lax.clamp(-1.0, jnp.linalg.norm(axis), 1.0))
    geodesicAngle = jax.lax.cond(jnp.dot(zUnit, newZ) > 0, lambda:theta, lambda:jnp.pi - theta)  
    return axis, geodesicAngle


def geodesicHopf_select_axis(newZ, planarAngle):
    # newZ should be a normalized vector
    # returns a 4x4 quaternion
    zUnit = jnp.array([0.0, 0.0, 1.0])

    # todo: implement cases where newZ is approx. -zUnit or approx. zUnit
    axis, geodesicAngle = jax.lax.cond(jnp.allclose(jnp.abs(newZ), zUnit, atol=1e-3), angle_axis_helper_edgecase, angle_axis_helper, newZ)

    return (transform_from_axis_angle(axis, geodesicAngle) @ transform_from_axis_angle(zUnit, planarAngle))


def fibonacci_sphere(samples_in_range, phi_range=jnp.pi):
    ga = jnp.pi * (jnp.sqrt(5) - 1)  # golden angle
    eps = 1e-10
    min_y = jnp.cos(phi_range)

    samples = jnp.round(samples_in_range * (2 / (1-min_y+eps)))

    def fib_point(i):
        y = 1 - (i / (samples - 1)) * 2  # goes from 1 to -1 
        radius = jnp.sqrt(1 - y * y)
        theta = ga * i
        x = jnp.cos(theta) * radius
        z = jnp.sin(theta) * radius
        return jnp.array([x,z,y])
        
    fib_sphere = jax.vmap(fib_point, in_axes=(0))
    points = jnp.arange(samples_in_range)
    return fib_sphere(points)


def get_rotation_proposals(fib_sample, rot_sample, min_rot_angle=0, max_rot_angle=2*jnp.pi, sphere_angle_range=jnp.pi):
    unit_sphere_directions = fibonacci_sphere(fib_sample, sphere_angle_range)
    geodesicHopf_select_axis_vmap = jax.vmap(jax.vmap(geodesicHopf_select_axis, in_axes=(0,None)), in_axes=(None,0))
    rot_stepsize = (max_rot_angle - min_rot_angle)/ rot_sample
    rotation_proposals = geodesicHopf_select_axis_vmap(unit_sphere_directions, jnp.arange(min_rot_angle, max_rot_angle, rot_stepsize)).reshape(-1, 4, 4)
    return rotation_proposals


def make_rotation_grid_enumeration(fibonacci_sphere_points, num_planar_angle_points, min_rot_angle=0, max_rot_angle=2*jnp.pi, sphere_angle_range=jnp.pi):
    return get_rotation_proposals(fibonacci_sphere_points, num_planar_angle_points, min_rot_angle, max_rot_angle, sphere_angle_range)


def make_translation_grid_enumeration(min_x,min_y,min_z, max_x,max_y,max_z, num_x=2,num_y=2,num_z=2):
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(min_x, max_x, num_x),
        jnp.linspace(min_y, max_y, num_y),
        jnp.linspace(min_z, max_z, num_z)
    ),
        axis=-1)
    deltas = deltas.reshape((-1,3),order='F')
    return jax.vmap(transform_from_pos)(deltas)

def make_translation_grid_enumeration_3d(min_x,min_y,min_z, max_x,max_y,max_z, num_x=2,num_y=2,num_z=2):
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(min_x,max_x,num_x),
        jnp.linspace(min_y,max_y,num_y),
        jnp.linspace(min_z,max_z,num_z),
    ),
        axis=-1)
    deltas = deltas.reshape(-1,3)
    return deltas

def make_translation_grid_enumeration_2d(min_x,min_y, max_x, max_y, num_x,num_y):
    deltas = jnp.stack(jnp.meshgrid(
        jnp.linspace(min_x,max_x,num_x),
        jnp.linspace(min_y,max_y,num_y),
    ),
        axis=-1)
    deltas = deltas.reshape(-1,2)
    return deltas


def make_grid_enumeration(min_x,min_y,min_z, min_rotation_angle, 
                        max_x,max_y,max_z, max_rotation_angle,
                        num_x,num_y,num_z, 
                        fibonacci_sphere_points, num_planar_angle_points, 
                        sphere_angle_range=jnp.pi):
    rotations = make_rotation_grid_enumeration(fibonacci_sphere_points, num_planar_angle_points, min_rotation_angle, max_rotation_angle, sphere_angle_range)
    translations = make_translation_grid_enumeration(min_x,min_y,min_z, max_x,max_y,max_z, num_x,num_y,num_z)
    all_proposals = jnp.einsum("aij,bjk->abik", rotations, translations).reshape(-1, 4, 4)
    return all_proposals
