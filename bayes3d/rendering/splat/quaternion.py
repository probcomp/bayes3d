# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""JAX-native quaternion utilities.
"""

import jax
import jax.numpy as jnp

Float = jax.Array
Quaternion = jax.Array
Vector3 = jax.Array
Matrix3 = jax.Array

_normalize = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
_split_and_squeeze = lambda x: (y.squeeze(-1) for y in jnp.split(x, 4, axis=-1))


def from_angle_and_vector(a: Float, v: Vector3) -> Quaternion:
  """Construct a quaternion from an angle and vector.

  The result is the quaternion

     cos(a / 2) + sin(a / 2) * (u[0] i + u[1] j + u[2] k).

  where u = v / |v| is the unit vector pointing in the direction of v.

  `a` and `v` can have non-trivial (broadcast-compatible) shape.

  Args:
    a: the angle of the implied rotation
    v: the vector pointing along the rotation axis.

  Returns:
    quaternion: the resulting quaternion as described above.
  """
  c, s = jnp.cos(a / 2), jnp.sin(a / 2)
  n = _normalize(v)
  return jnp.concatenate([
      c * jnp.ones(n.shape[:-1] + (1,)),
      s * n
  ], axis=-1)


def pure_imag(v: Vector3) -> Quaternion:
  """Create a pure imaginary quaternion from a 3-vector.

  The resulting quaternion is

    v[0] i + v[1] j + v[2] k

  Args:
    v: vector input

  Returns:
    quaternion: the pure imaginary quaternion with imag. components from v.
  """
  return jnp.concatenate([jnp.zeros(v.shape[:-1] + (1,)), v], axis=-1)


def quatmul(a: Quaternion, b: Quaternion) -> Quaternion:
  """Multiply two quaternions.

  Args:
    a: first quaternion to multiply.
    b: second quaternion to multiply.

  Returns:
    quaternion: the quaternion product of a with b.
  """
  a0, a1, a2, a3 = _split_and_squeeze(a)
  b0, b1, b2, b3 = _split_and_squeeze(b)
  return jnp.stack([
      a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
      a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
      a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
      a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
  ], axis=-1)


def quatvec(q: Quaternion, v: Vector3) -> Vector3:
  """Apply a quaternion to a vector.

  This is q * v' * conjugate(q), where v' is the pure imaginary quaternion with
  imaginary components from v, and the multiplication is quaternion
  multiplication.

  When q is a unit quaternion, the effect is rotation of angle a of v about an
  axis pointing in direction u, where

     q = cos(a / 2) + sin(a / 2) * (u[0] i + u[1] j + u[2] k).

  When q is not a unit quaternion, the effect is a rotation and scaling of v.

  Args:
    q: the quaternion to apply to v.
    v: the 3-vector to apply q to.

  Returns:
    vector: the vector resulting from application of q to v.
  """
  # Below should be faster than naive conjugation as in
  #   return quatmul(q, quatmul(pure_imag(v), conjugate(q)))[..., 1:]
  # Why? Each quatmul is 16 multiplies, 12 adds, as well as slice and stack
  # operations (potentially costing copies/mem reorg). This totals
  #   32 mul, 24 add, 2 slice, 2 stack
  #
  # Cross product of (p1, p2, p3) and (q1, q2, q3) is
  #   (p2 * q3 - p3 * q2,
  #    p3 * q1 - p1 * q3,
  #    p1 * q2 - p2 * q1)
  # which is 6 multiplies and 3 adds. We do this twice, along with a few more of
  # each, for a total of maybe 14 mul, 8 add, 2 slice. The crosses also
  # presumably incur some slices! Still, this should be more efficient. Take
  # from eigen forum: https://eigen.tuxfamily.org/bz/show_bug.cgi?id=1779
  #
  # NB, as noted in the eigen code, conversion to 3x3 matrix and subsequent
  # rotation is more performant for more than one vector.
  n = norm(q, keepdims=True)
  qw = q[..., :1]
  qxyz = q[..., 1:]
  t = 2 * jnp.cross(qxyz, v)
  return n ** 2 * v + qw * t + jnp.cross(qxyz, t)


def norm(q: Quaternion, keepdims: bool = False) -> Float:
  """Compute the 2-norm of quaternion q.

  This is the square root of the sum of the squares of the components of q.

  Args:
    q: quaternion whose norm to compute.
    keepdims: if True, retain the innermost dimension when summing over
        components of q.

  Returns:
    norm: the 2-norm of q.
  """
  return jnp.sqrt(inner_prod(q, q, keepdims=keepdims))


def conjugate(q: Quaternion) -> Quaternion:
  """Conjugate the quaternion q.

  This means negating the imaginary components, leaving the real one fixed.

  Args:
    q: quaternion

  Returns:
    conj: the conjugate of q.
  """
  real = q[..., :1]
  imag = -q[..., 1:]
  return jnp.concatenate([real, imag], axis=-1)


def inner_prod(q: Quaternion, p: Quaternion, keepdims: bool = False) -> Float:
  """Compute the inner product of two quaternions.

  This is the sum of the products of the components of q and conj(p). That is,
  if
    q = q0 + q1 i + q2 j + q3 k
    p = p0 + p1 i + p2 j + p3 k

  then
    q . p = q0 p0 + q1 i p1 (-i)+ q2 j p2 (-j)  + q3 k p3 (-k)
    q . p = q0 p0 + q1 p1 (-i i) + q2 p2 (-j j)  + q3 p3 (-k k)
          = q0 p0 + q1 p1 + q2 p2 + q3 p3

  Args:
    q: first quaternion
    p: second quaternion
    keepdims: if True, retain the innermost dimension when summing over
        components of p * q.

  Returns:
    inner_prod: the inner product of q with p.
  """
  return (q * p).sum(axis=-1, keepdims=keepdims)


def inverse(q: Quaternion) -> Quaternion:
  """Compute the inverse of the quaternion q.

  This is the conjugate of q, divided by its squared norm.

  Args:
    q: the quaternion to invert

  Returns:
    inverse: the inverse of q.
  """
  return conjugate(q) / inner_prod(q, q, keepdims=True)


def to_matrix(q: Quaternion) -> Matrix3:
  """Compute the matrix that has the same effect as applying q to a 3-vector.

  That is, the matrix m whose product (m @ v) with a vector v, is the same
  vector as quatvec(q, v) := q * v * conjugate(q).

  Args:
    q: quaternion to convert to matrix form

  Returns:
    m: the matrix form of q.
  """
  perm = list(range(len(q.shape) + 1))
  perm[-2], perm[-1] = perm[-1], perm[-2]
  return quatvec(q[..., jnp.newaxis, :], jnp.eye(3)).transpose(perm)
