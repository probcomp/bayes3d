import jax
import jax.numpy as jnp


def transform_from_pos(t):
    return jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_rot(rot):
    return jnp.vstack(
        [jnp.hstack([rot, jnp.zeros(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_rot_and_pos(rot, t):
    return jnp.vstack(
        [jnp.hstack([rot, t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_axis_angle(axis, angle):
    sina = jnp.sin(angle)
    cosa = jnp.cos(angle)
    direction = axis / jnp.linalg.norm(axis)
    # rotation matrix around unit vector
    R = jnp.diag(jnp.array([cosa, cosa, cosa]))
    R = R + jnp.outer(direction, direction) * (1.0 - cosa)
    direction = direction * sina
    R = R + jnp.array([[0.0, -direction[2], direction[1]],
                        [direction[2], 0.0, -direction[0]],
                        [-direction[1], direction[0], 0.0]])
    M = jnp.identity(4)
    M = M.at[:3, :3].set(R)
    return M

# move point cloud to a specified pose
# coords: (N,3) point cloud
# pose: (4,4) pose matrix. rotation matrix in top left (3,3) and translation in (:3,3)
def apply_transform(coords, transform):
    coords = jnp.einsum(
        'ij,...j->...i',
        transform,
        jnp.concatenate([coords, jnp.ones(coords.shape[:-1] + (1,))], axis=-1),
    )[..., :-1]
    return coords

def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = jnp.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


@jax.jit
def angle_axis_helper_edgecase(newZ):
    zUnit = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.array([0.0, 1.0, 0.0])
    geodesicAngle = jax.lax.cond(jnp.allclose(newZ, zUnit, atol=1e-3), lambda:0.0, lambda:jnp.pi)
    return axis, geodesicAngle


@jax.jit
def angle_axis_helper(newZ): 
    zUnit = jnp.array([1.0, 0.0, 0.0])
    axis = jnp.cross(zUnit, newZ)
    theta = jax.lax.asin(jax.lax.clamp(-1.0, jnp.linalg.norm(axis), 1.0))
    geodesicAngle = jax.lax.cond(jnp.dot(zUnit, newZ) > 0, lambda:theta, lambda:jnp.pi - theta)  
    return axis, geodesicAngle


@jax.jit
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
        
    fib_sphere = jax.jit(jax.vmap(fib_point, in_axes=(0)))
    return fib_sphere(jnp.arange(samples))