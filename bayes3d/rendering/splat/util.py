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
"""Utilities for the splat library.

Generating random data, et cetera.
"""

import jax
from jax import numpy as jnp

from . import types
from . import quaternion


Array = types.Array
Key = types.Key
Blob = types.Blob
Splat = types.Splat
Bounds = types.Bounds
Grid = types.Grid


def generate_random_blob(key: Key,
                         *,
                         sph_harm_dim: int = 4,
                         ) -> Blob:
  """Generate a random Blob."""
  assert sph_harm_dim in (1, 4, 9, 16, 25), sph_harm_dim
  keys = jax.random.split(key, 6)
  xyz = jax.random.normal(keys[0], (3,)) + jnp.array([0, 0, 2.])
  eigvals = jax.random.uniform(keys[1], (3,), minval=.5, maxval=1.) * .0005
  angle = jax.random.uniform(keys[2], (), minval=-jnp.pi, maxval=jnp.pi)
  vector = jax.random.normal(keys[3], (3,))
  vector /= jnp.linalg.norm(vector, ord=2) + 1e-6
  rot = quaternion.to_matrix(quaternion.from_angle_and_vector(angle, vector))
  cov = rot @ jnp.diag(eigvals) @ rot.T
  color = jax.random.normal(keys[4], (sph_harm_dim, 3,))
  alpha = jax.nn.sigmoid(jax.random.normal(keys[5]))
  return Blob(xyz=xyz, cov=cov, color=color, alpha=alpha)


def make_image_plane(grid: Grid) -> tuple[Array, float]:
  """Compute grid of pixel coordinates from a Grid."""
  (xmin, xmax, ymin, ymax), xres, yres = grid
  xsize = (xmax - xmin) / xres
  ysize = (ymax - ymin) / yres
  xs = jnp.linspace(xmin, xmax, xres, endpoint=False) + (xsize * .5)
  ys = jnp.linspace(ymin, ymax, yres, endpoint=False) + (ysize * .5)
  pixel_area = xsize * ysize
  xs, ys = jnp.meshgrid(xs, ys, indexing='xy')
  xys = jnp.stack([xs, ys], axis=-1)
  return xys, pixel_area


def splat_inlier(splat: Splat, bounds: Bounds) -> bool:
  """Returns whether the center of the splat is within 2x of the bounds."""
  if splat.xyd.ndim > 1:
    return jax.vmap(lambda s: splat_inlier(s, bounds))(splat)
  x, y, d = splat.xyd
  minx, maxx, miny, maxy = bounds
  xcenter = (maxx + minx) / 2
  dx = maxx - minx
  ycenter = (maxy + miny) / 2
  dy = maxy - miny
  return ((x > xcenter - dx) & (x < xcenter + dx) &
          (y > ycenter - dy) & (y < ycenter + dy) &
          # TODO: Figure out d(x,y) at image plane
          (d > .2))


def in_frustum(splat: Splat, bounds: Bounds) -> bool:
  """Returns whether the splat radius is contained within the bounds."""
  if splat.xyd.ndim > 1:
    return jax.vmap(lambda s: in_frustum(s, bounds))(splat)
  x, y, d = splat.xyd
  # TODO: Consider radius == C * max eigval or 99 percentile.
  return (# (x > bounds.minx - splat.radius * 3) &
  #         (x < bounds.maxx + splat.radius * 3) &
  #         (y > bounds.miny - splat.radius * 3) &
  #         (y < bounds.maxy + splat.radius * 3) &
  #         # TODO: Is the below correct?
  #         (splat.radius < jnp.minimum(bounds.maxx - bounds.minx,
  #                                     bounds.maxy - bounds.miny)) &
          # TODO: Figure out d(x,y) at image plane
          (d > .2))


@jax.jit
def tile_idx(xy, grid):
  dx = (grid.bounds.maxx - grid.bounds.minx) / grid.nx
  tx = ((xy[0] - grid.bounds.minx) / dx).astype(jnp.int32)
  dy = (grid.bounds.maxy - grid.bounds.miny) / grid.ny
  ty = ((xy[1] - grid.bounds.miny) / dy).astype(jnp.int32)
  return jnp.clip(tx, 0, grid.nx), jnp.clip(ty, 0, grid.ny)


@jax.jit
def get_tile_rect(xy, radius, grid):
  min_tile = tile_idx(xy - 3 * radius, grid)
  dx = (grid.bounds.maxx - grid.bounds.minx) / grid.nx
  dy = (grid.bounds.maxy - grid.bounds.miny) / grid.ny
  max_tile = tile_idx(xy + 3 * radius + jnp.float32([dx, dy]), grid)
  return min_tile, max_tile


@jax.jit
def count_touched_tiles(xy, rad, grid):
  min_tile, max_tile = get_tile_rect(xy, rad, grid)
  return (max_tile[0] - min_tile[0]) * (max_tile[1] - min_tile[1])


# The hardcoded values for Spherical Harmonics functions were taken from
# https://github.com/google/spherical-harmonics


def _spherical_harmonics_0(sh_coeff, directions):
  """Hardcoded spherical harmonics of degree 0."""
  sh_0 = 0.28209479177387814 * sh_coeff[..., 0, :] * jnp.ones_like(directions)
  return sh_0


def _spherical_harmonics_1(sh_coeff, directions):
  """Hardcoded spherical harmonics of degree 1."""
  sh_0 = _spherical_harmonics_0(sh_coeff, directions)
  x, y, z = jnp.split(directions, 3, axis=-1)
  sh_1 = (
      sh_0
      - 0.4886025119029199 * y * sh_coeff[..., 1, :]
      + 0.4886025119029199 * z * sh_coeff[..., 2, :]
      - 0.4886025119029199 * x * sh_coeff[..., 3, :]
  )
  return sh_1


def _spherical_harmonics_2(sh_coeff, directions):
  """Hardcoded spherical harmonics of degree 2."""
  sh_1 = _spherical_harmonics_1(sh_coeff, directions)
  x, y, z = jnp.split(directions, 3, axis=-1)
  xx, yy, zz = x * x, y * y, z * z
  xy, yz, xz = x * y, y * z, x * z

  sh_2 = (
      sh_1
      + 1.0925484305920792 * xy * sh_coeff[..., 4, :]
      + -1.0925484305920792 * yz * sh_coeff[..., 5, :]
      + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh_coeff[..., 6, :]
      + -1.0925484305920792 * xz * sh_coeff[..., 7, :]
      + 0.5462742152960396 * (xx - yy) * sh_coeff[..., 8, :]
  )
  return sh_2


def _spherical_harmonics_3(sh_coeff, directions):
  """Hardcoded spherical harmonics of degree 3."""
  sh_2 = _spherical_harmonics_2(sh_coeff, directions)
  x, y, z = jnp.split(directions, 3, axis=-1)
  xx, yy, zz = x * x, y * y, z * z
  xy = x * y

  sh_3 = (
      sh_2
      + -0.5900435899266435 * y * (3 * xx - yy) * sh_coeff[..., 9, :]
      + 2.890611442640554 * xy * z * sh_coeff[..., 10, :]
      + -0.4570457994644658 * y * (4 * zz - xx - yy) * sh_coeff[..., 11, :]
      + 0.3731763325901154
      * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeff[..., 12, :]
      + -0.4570457994644658 * x * (4 * zz - xx - yy) * sh_coeff[..., 13, :]
      + 1.445305721320277 * z * (xx - yy) * sh_coeff[..., 14, :]
      + -0.5900435899266435 * x * (xx - 3 * yy) * sh_coeff[..., 15, :]
  )
  return sh_3


def _spherical_harmonics_4(sh_coeff, directions):
  """Hardcoded spherical harmonics of degree 4."""
  sh_3 = _spherical_harmonics_3(sh_coeff, directions)
  x, y, z = jnp.split(directions, 3, axis=-1)
  xx, yy, zz = x * x, y * y, z * z
  xy, yz, xz = x * y, y * z, x * z
  sh_4 = (
      sh_3
      + 2.5033429417967046 * xy * (xx - yy) * sh_coeff[..., 16, :]
      + -1.7701307697799304 * yz * (3 * xx - yy) * sh_coeff[..., 17, :]
      + 0.9461746957575601 * xy * (7 * zz - 1) * sh_coeff[..., 18, :]
      + -0.6690465435572892 * yz * (7 * zz - 3) * sh_coeff[..., 19, :]
      + 0.10578554691520431 * (zz * (35 * zz - 30) + 3) * sh_coeff[..., 20, :]
      + -0.6690465435572892 * xz * (7 * zz - 3) * sh_coeff[..., 21, :]
      + 0.47308734787878004 * (xx - yy) * (7 * zz - 1) * sh_coeff[..., 22, :]
      + -1.7701307697799304 * xz * (xx - 3 * yy) * sh_coeff[..., 23, :]
      + 0.6258357354491761
      * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh_coeff[..., 24, :]
  )
  return sh_4


@jax.jit
def apply_sh(sh_coeff: Array, directions: Array) -> Array:
  """Applies spherical harmonics for a given direction[s]."""
  cases = {
    1:
      lambda: _spherical_harmonics_0(sh_coeff, directions),
    4:
      lambda: _spherical_harmonics_1(sh_coeff, directions),
    9:
      lambda: _spherical_harmonics_2(sh_coeff, directions),
    16:
      lambda: _spherical_harmonics_3(sh_coeff, directions),
    25:
      lambda: _spherical_harmonics_4(sh_coeff, directions),
  }
  return cases[sh_coeff.shape[-2]]()
