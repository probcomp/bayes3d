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
"""3D Gaussian splatting.

https://arxiv.org/abs/2308.04079
"""
import functools
from typing import Any, Mapping, NamedTuple
from types import SimpleNamespace

from flax.core import frozen_dict
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import jax.tree_util as jtu

# from . import gpu_sort
gpu_sort = SimpleNamespace(sort_pairs=jax.lax.sort_key_val)
from . import types
from . import util


Array = types.Array
Blob = types.Blob
Camera = types.Camera
Grid = types.Grid
Splat = types.Splat
Pytree = Any

FrozenDict = frozen_dict.FrozenDict


# EWA splatting from Zwicker et al 2001.
#
# We assume a world-to-camera coordinate transformation has been applied, so we
# have camera coordinates (x_c, y_c, z) with the camera at the origin and the
# viewing plane at  z = 1.  We first transform from camera space to "ray space".
# Ray space is parameterized by (x_r, y_r, d) where x_r, y_r is the image
# plane coordinate of the ray-image-plane intersection, and d is the distance
# along this ray.
#
# Given (x_c, y_c, z) in camera coordinates, the distance along the ray is the
# norm, and we scale $x_c, y_c$ by $1/z$ to get the image plane coordinates.
#
# x_r = x_c / z
# y_r = y_c / z
# d = sign(z) * sqrt(x_c^2 + y_c^2 + z^2)
#
# Inverting this gives
#
# d^2 = x_c^2 + y_c^2 + z^2
#     = z^2(x_r^2 + y_r^2 + 1)
# =>   z = d / sqrt(x_r^2 + y_r^2 + 1)
#    x_c = x_r * z
#    y_c = y_r * z


def camera_to_ray(xyz: Array,
                  camera: Camera) -> Array:
  """Convert camera-space xyz to image-space xy and depth."""
  x_c, y_c, z = xyz
  x_r = x_c / (z + 1e-7) * camera.fx
  y_r = y_c / (z + 1e-7) * camera.fy
  norm = jnp.linalg.norm(xyz) * jnp.sign(z)
  return jnp.array([x_r, y_r, norm])


def camera_to_ray_no_depth(xyz: Array,
                           camera: Camera) -> Array:
  """Convert camera-space xyz to image-space xy."""
  return xyz[:2] / (xyz[2] + 1e-7) * jnp.array([camera.fx, camera.fy])


# Projecting a Gaussian onto an image plane is non-linear, and hence
# non-Gaussian.  Zwicker et al approximate this by linearizing the projection of
# the covariance matrix.  We manually compute this for efficiency.


def projection_jacobian(xyz: Array,
                        camera: Camera) -> Array:
  """Computes jax.jacobian(camera_to_ray_no_depth)(xyz)."""
  x, y, z = xyz
  fx, fy = camera.fx, camera.fy
  w, h = camera.width, camera.height
  tan_fov_x, tan_fov_y = w / (2 * fx), h / (2 * fy)
  z = jnp.maximum(1e-7, jnp.abs(z)) * jnp.sign(z)
  x = z * jnp.clip(x / z, -1.3 * tan_fov_x, 1.3 * tan_fov_x)
  y = z * jnp.clip(y / z, -1.3 * tan_fov_y, 1.3 * tan_fov_y)
  zm1 = 1 / z
  zm2 = zm1 * zm1
  return jnp.array([[fx * zm1, 0., -fx * x * zm2],
                    [0., fy * zm1, -fy * y * zm2]])


def make_sandwich(bread: Array, meat: Array) -> Array:
  """Compute B @ M @ B.T where M is symmetric."""
  b0, b1 = bread
  m0, m1 = bread @ meat
  offdiag = m0 @ b1
  return jnp.array(
      [[m0 @ b0, offdiag],
       [offdiag, m1 @ b1]])


def projection_jacobian_view_product(
    xyz: Array,
    camera: Camera) -> Array:
  """Computes jax.jacobian(camera_to_ray_no_depth)(xyz) @ view."""
  x, y, z = xyz
  fx, fy = camera.fx, camera.fy
  w, h = camera.width, camera.height
  tan_fov_x, tan_fov_y = w / (2 * fx), h / (2 * fy)
  z = jnp.maximum(1e-7, jnp.abs(z)) * jnp.sign(z)
  x = jnp.clip(x / z, -1.3 * tan_fov_x, 1.3 * tan_fov_x) * z
  y = jnp.clip(y / z, -1.3 * tan_fov_y, 1.3 * tan_fov_y) * z
  ((w00, w01, w02),
   (w10, w11, w12),
   (w20, w21, w22)) = camera.orientation
  zm1 = 1 / (z + 1e-7)
  w20zm1 = w20 * zm1
  w21zm1 = w21 * zm1
  w22zm1 = w22 * zm1

  # fx, fy = 1, 1  # TODO: Why is rendering better when we ignore fx,fy?
  return zm1 * jnp.array(
      [fx * jnp.array([w00 - w20zm1 * x, w01 - w21zm1 * x, w02 - w22zm1 * x]),
       fy * jnp.array([w10 - w20zm1 * y, w11 - w21zm1 * y, w12 - w22zm1 * y])])


def max_eigval_2x2(mat: Array) -> Array:
  """Computes radius 9ms faster (@1M splats) than jnp.eigvalsh(mat).max()."""
  ((a, b), (c, d)) = mat
  tr = a + d
  det = a * d - b * c
  return jax.nn.relu(tr / 2) + jnp.sqrt(jax.nn.relu(tr**2 / 4 - det))


@functools.partial(jax.jit, static_argnames=('color_clipping',))
def splat_blob(blob: Blob,
               camera: Camera,
               color_clipping: str = 'linear_clip',
               ) -> Splat:
  """World-to-camera transform blob and splat onto the image plane (z = 1).

  Args:
    blob: Gaussians to splat into 2d.
    camera: Camera intrinsics and extrinsics.
    color_clipping: Color clipping behavior. Options are 'sigmoid' or
        'linear_clip'.

  Returns:
    splat: `types.Splat` splatted blobs.
  """
  if blob.xyz.ndim > 1:
    return jax.vmap(
        functools.partial(splat_blob, color_clipping=color_clipping),
        in_axes=(0, None))(blob, camera)
  # Suppose we have a 3D Gaussian blob with mean `m` and covariance `S`.
  # We have a world-to-camera transformation
  #   t := V(u) := R @ u + d
  # (where R is orthogonal and d is a translation) and the (non-affine)
  # camera-to-ray projection
  #   x := f(t) := [t0 / t2, t1 / t2, norm(t)].T
  # Define the composite g(u) := f(V(u)).  The first order Taylor approximation
  # of g around the mean `m` of a blob is
  #   g(u) ~ g(m) + J_g(m) @ (u - m)
  # where J_g(m) is the Jacobian of g.  Note J_g(m) = J_f(V(m)) @ R.
  # Then the push-forward of the blob is approximately a 2D Gaussian
  # with mean `g(m)` and covariance `J_f(V(m)) @ R @ S @ (J_f(V(m)) @ R).T`.
  cam = (blob.xyz - camera.xyz) @ camera.orientation.T
  # i.e. (camera.orientation @ (blob.xyz - camera.xyz).T).T
  bread = projection_jacobian_view_product(cam, camera)
  splat_xyd = camera_to_ray(cam, camera)
  splat_cov = make_sandwich(bread, blob.cov) + jnp.eye(2) * 0.3  # low pass

  # Compute spherical harmonics.
  dirn = blob.xyz - camera.xyz
  dirn = jnp.where((dirn == 0.).all(), 1., dirn)
  dirn = dirn / jnp.linalg.norm(dirn, ord=2, axis=-1)
  color = util.apply_sh(blob.color, dirn)
  if color_clipping == 'linear_clip':
    color = jnp.clip(color + .5, 0., 1.)
  elif color_clipping == 'sigmoid':
    color = jax.nn.sigmoid(color)
  else:
    raise ValueError(f'Unexpected `color_clipping` arg: {color_clipping}.')

  splat_rad = jnp.sqrt(max_eigval_2x2(splat_cov))
  return Splat(
      xyd=splat_xyd,
      cov=splat_cov,
      radius=splat_rad,
      color=color,
      alpha=blob.alpha)


TRANSMITTANCE_MIN = 0.01


def render_sorted_splats(
    xy: jax.Array,
    sorted_splats: Splat,
    unroll: int = 25,
) -> jax.Array:
  """Render splats at an image-plane pixel."""

  def cont_condition(i, arg):
    log_transmittance, _ = arg
    return ((log_transmittance > jnp.log(TRANSMITTANCE_MIN)) &
            (i < sorted_splats.xyd.shape[0]))

  @jax.jit
  @jax.remat
  def body(i, arg):
    log_transmittance, color = arg
    splat = jtu.tree_map(lambda x: x[i], sorted_splats)
    # TODO: Optimize via manual solve.
    solve_x = jnp.linalg.solve(splat.cov, xy - splat.xyd[:2])
    # TODO: Write this out manually and benchmark.
    density = jnp.exp(-.5 * jnp.dot(xy - splat.xyd[:2], solve_x))
    alpha = density * splat.alpha
    still_valid = cont_condition(i, arg)
    color += jnp.where(
        still_valid, jnp.exp(log_transmittance) * alpha * splat.color, 0.)
    log_transmittance += jnp.log1p(-alpha)
    return log_transmittance, color

  @jax.remat
  def unrolled_body(i, carry):
    i = i * unroll
    for _ in range(unroll):
      carry = jax.lax.cond(
          cont_condition(i, carry),
          functools.partial(body, i, carry),
          lambda: carry)
      i += 1
    return carry

  log_transmittance = jnp.zeros(())
  color = jnp.zeros(3)
  _, color = jax.lax.fori_loop(  # use fori_loop to get backprop
      0, (sorted_splats.xyd.shape[0] + unroll - 1) // unroll,
      unrolled_body, (log_transmittance, color))
  return color


def _unchanged_block_spec(arr):
  if isinstance(arr, jax.Array):
    return pl.BlockSpec(lambda *_: (0,) * arr.ndim, arr.shape)
  if isinstance(arr, tuple):
    return pl.BlockSpec(lambda *_: (0,) * len(arr), arr)
  raise ValueError


### Pallas forward renderer.


class Tile(NamedTuple):
  xy: jax.Array
  work_start: jax.Array
  work_limit: jax.Array


def _solve_2d(A, xy):  # pylint: disable=invalid-name
  """Unpacks 2x2 tuple A and 2-tuple xy and solves the 2D system."""
  (a, b), (c, d) = A
  x, y = xy
  # LU = A, where
  # L = [[1, 0], [e, 1]]
  # U = [[f, g], [0, h]]
  f = a
  g = b
  e = c / a
  h = d - b * e
  # backwards solve Lwz = xy
  w = x
  z = y - e * x
  # forwards solve Uuv = wz
  v = z / h
  u = (w - g * v) / f
  return [u, v]


BLOCK_X = 16
BLOCK_Y = 16


@functools.partial(jax.jit, static_argnames=('return_extras', 'pallas_kwargs'))
def _render_img_pl_wrapper(xys, splats, tile_work_bounds, work_splat_idx,
                           return_extras=False, pallas_kwargs=None):
  """JITs a call to a pallas-triton forward rendering kernel."""
  img_y, img_x, _ = xys.shape
  img_shp = img_y, img_x
  ny, nx = (img_y + BLOCK_Y - 1) // BLOCK_Y, (img_x + BLOCK_X - 1) // BLOCK_X

  # If we are only renderering a forward pass, we don't require extra outputs,
  # so we'll avoid computing them.
  out_shape = (
      jax.ShapeDtypeStruct((*img_shp, 4), jnp.float32),  # color
  )
  out_specs = (
      pl.BlockSpec(lambda i, j: (i, j, 0), (BLOCK_Y, BLOCK_X, 4)),  # color
  )
  if return_extras:
    out_shape += (
        jax.ShapeDtypeStruct(img_shp, jnp.int32),  # final_count
        jax.ShapeDtypeStruct(img_shp, jnp.float32),  # final_transmittance
    )
    out_specs += (
        pl.BlockSpec(lambda i, j: (i, j), (BLOCK_Y, BLOCK_X)),  # final_count
        pl.BlockSpec(lambda i, j: (i, j),
                     (BLOCK_Y, BLOCK_X)),  # final_transmittance
    )

  @functools.partial(
      pl.pallas_call,
      out_shape=out_shape,
      in_specs=(
          Tile(xy=pl.BlockSpec(lambda i, j: (i, j, 0), (BLOCK_Y, BLOCK_X, 2)),
               # TODO: Why can't we use a scalar shape for these?
               work_start=pl.BlockSpec(lambda i, j: (i, j), (1, 1)),
               work_limit=pl.BlockSpec(lambda i, j: (i, j), (1, 1))),
          # the rest of the args are the same across all blocks
          Splat(xyd=_unchanged_block_spec(splats.xyd),
                cov=_unchanged_block_spec(splats.cov),
                alpha=_unchanged_block_spec(splats.alpha),
                color=_unchanged_block_spec((splats.color.shape[0], 4)),
                radius=()),
          _unchanged_block_spec(work_splat_idx),  # work_splat_idx
      ),
      out_specs=out_specs,
      grid=(ny, nx),
      **(pallas_kwargs or {}))
  def _render_img_pallas(
      tile: Tile,  # unique per block
      splats: Splat,  # shared across blocks
      work_splat_idx: jax.Array,  # shared across blocks
      color_out: jax.Array,  # rendered tile image
      *extras_out,
      ):
    # Each block does a different amount of work, delimited by work_start and
    # work_limit. This work range identifies a tile-specific, sorted-by-distance
    # range of splat indices from `work_splat_idx`.
    if return_extras:
      extras_out: tuple[jax.Array, jax.Array]
      final_count_out, final_transmittance_out = extras_out
    else:
      () = extras_out
      final_count_out, final_transmittance_out = None, None

    work_start, work_limit = tile.work_start[0, 0], tile.work_limit[0, 0]

    def cond(arg):
      log_transmittance, _, _, i = arg
      # TODO: .any() unsupported by pallas-triton
      return (((log_transmittance > jnp.log(TRANSMITTANCE_MIN)).sum() > 0) &
              (i < work_limit))

    def body(arg):
      log_transmittance, color, final_count, i = arg
      splat_idx = work_splat_idx[i]
      cov = ((splats.cov[splat_idx, 0, 0], splats.cov[splat_idx, 0, 1]),
             (splats.cov[splat_idx, 1, 0], splats.cov[splat_idx, 1, 1]))
      x, y = splats.xyd[splat_idx, 0], splats.xyd[splat_idx, 1]
      centered_xy = (tile.xy[:, :, 0] - x, tile.xy[:, :, 1] - y)
      solve_xy = _solve_2d(cov, centered_xy)
      dot_out = centered_xy[0] * solve_xy[0] + centered_xy[1] * solve_xy[1]
      density = jnp.exp(-.5 * dot_out)
      still_valid = ((log_transmittance > jnp.log(TRANSMITTANCE_MIN)) &
                     (i < work_limit))
      alpha = jnp.where(still_valid, density * splats.alpha[splat_idx], 0.)
      final_count += jnp.where(still_valid, 1, 0)
      color += ((jnp.exp(log_transmittance) * alpha)[..., None] *
                splats.color[splat_idx, :])
      log_transmittance += jnp.log1p(-alpha)
      return (jnp.float32(log_transmittance), jnp.float32(color), final_count,
              i + 1)

    # Initialize state and outputs.
    log_transmittance = jnp.zeros(tile.xy.shape[:-1], jnp.float32)
    color = jnp.zeros(tile.xy.shape[:-1] + (4,), jnp.float32)
    final_count = jnp.zeros(tile.xy.shape[:-1], jnp.int32)

    log_transmittance, color, final_count, _ = jax.lax.while_loop(
        cond, body, (log_transmittance, color, final_count, work_start))

    color_out[:] = jnp.float32(color)
    if return_extras:
      final_count_out[:] = final_count
      final_transmittance_out[:] = jnp.float32(log_transmittance)

  # Actually call the pallas kernel.
  im, *extras = _render_img_pallas(  # pylint:disable=no-value-for-parameter
      Tile(xy=xys,
           work_start=tile_work_bounds[:-1].reshape(ny, nx),
           work_limit=tile_work_bounds[1:].reshape(ny, nx)),
      splats._replace(radius=(), color=jnp.pad(splats.color, ((0, 0), (0, 1)))),
      work_splat_idx)
  if return_extras:
    args = xys, splats, tile_work_bounds, work_splat_idx
    return im[..., :3], (args, *extras)
  return im[..., :3]


### Pallas backprop kernel


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _render_img(pallas_kwargs, xys, splats, tile_work_bounds, work_splat_idx):
  return _render_img_pl_wrapper(xys, splats, tile_work_bounds, work_splat_idx,
                                pallas_kwargs=pallas_kwargs)


def _render_img_fwd(pl_kwargs, xys, splats, tile_work_bounds, work_splat_idx):
  return _render_img_pl_wrapper(xys, splats, tile_work_bounds, work_splat_idx,
                                return_extras=True, pallas_kwargs=pl_kwargs)


def _render_img_bwd(pallas_kwargs: Mapping[str, Any],
                    res: tuple[Pytree, jax.Array, jax.Array],
                    dfinal_color: jax.Array):
  """Computes the backward pass for the per-tile splat render."""
  args, final_count, final_transmittance = res
  # args wrt which we compute grads:
  xys, splats, tile_work_bounds, work_splat_idx = args

  img_y, img_x, *_ = xys.shape
  ny, nx = (img_y + BLOCK_Y - 1) // BLOCK_Y, (img_x + BLOCK_X - 1) // BLOCK_X

  workspace_dim = work_splat_idx.shape[0]

  # First, we will compute the gradient for the work unit, then segment-sum
  # these. Alternatively, we could atomic_add the gradients for a given splat,
  # which might be more performant. There are opportunities to ameliorate
  # numerical issues with segment_sum, e.g. with bucket_size=N whereas
  # atomic_add just commits to numerical instability.
  @functools.partial(
      pl.pallas_call,
      out_shape=(
          Tile(  # dtile
              xy=jax.ShapeDtypeStruct(xys.shape, jnp.float32),
              work_start=(), work_limit=()),
          Splat(  # dsplats
              xyd=jax.ShapeDtypeStruct((workspace_dim, 3), jnp.float32),
              cov=jax.ShapeDtypeStruct((workspace_dim, 2, 2), jnp.float32),
              alpha=jax.ShapeDtypeStruct((workspace_dim,), jnp.float32),
              color=jax.ShapeDtypeStruct((workspace_dim, 4), jnp.float32),
              radius=()),
      ),
      in_specs=(
          Splat(xyd=_unchanged_block_spec((workspace_dim, 3)),
                cov=_unchanged_block_spec((workspace_dim, 2, 2)),
                alpha=_unchanged_block_spec((workspace_dim,)),
                color=_unchanged_block_spec((workspace_dim, 4)),
                radius=()),  # dsplats initializer
          Tile(xy=pl.BlockSpec(lambda i, j: (i, j, 0), (BLOCK_Y, BLOCK_X, 2)),
               # TODO: Why can't we use a scalar shape for these?
               work_start=pl.BlockSpec(lambda i, j: (i, j), (1, 1)),
               work_limit=pl.BlockSpec(lambda i, j: (i, j), (1, 1))),
          Splat(xyd=_unchanged_block_spec(splats.xyd),  # same across all blocks
                cov=_unchanged_block_spec(splats.cov),
                alpha=_unchanged_block_spec(splats.alpha),
                color=_unchanged_block_spec((splats.color.shape[0], 4)),
                radius=()),
          _unchanged_block_spec(work_splat_idx),  # same across all blocks
          pl.BlockSpec(lambda i, j: (i, j), (BLOCK_Y, BLOCK_X)),  # final_count
          pl.BlockSpec(lambda i, j: (i, j),
                       (BLOCK_Y, BLOCK_X)),  # final_transmittance
          pl.BlockSpec(lambda i, j: (i, j, 0),
                       (BLOCK_Y, BLOCK_X, 4)),  # dfinal_color
      ),
      out_specs=(
          Tile(xy=pl.BlockSpec(lambda i, j: (i, j, 0), (BLOCK_Y, BLOCK_X, 2)),
               work_start=(), work_limit=()),
          Splat(xyd=_unchanged_block_spec((workspace_dim, 3)),
                cov=_unchanged_block_spec((workspace_dim, 2, 2)),
                alpha=_unchanged_block_spec((workspace_dim,)),
                color=_unchanged_block_spec((workspace_dim, 4)),
                radius=()),
      ),
      input_output_aliases={0: 1, 1: 2, 2: 3, 3: 4},  # dsplats initializer
      grid=(ny, nx),
      **(pallas_kwargs or {}),
      )
  def _bwd_render_img_pallas(dsplats_initializer: Splat,
                             tile: Tile,
                             splats: Splat,
                             work_splat_idx: jax.Array,
                             final_count: jax.Array,
                             final_transmittance: jax.Array,
                             dfinal_color: jax.Array,  # input
                             dtile: Tile,  # output
                             dsplats: Splat):  # output
    del dsplats_initializer  # used for zero-init
    log_transmittance = final_transmittance[:]
    dfinal_color = dfinal_color[:]
    work_start = tile.work_start[0, 0]
    final_count = final_count[:, :]

    def body(_, arg):
      dxy, log_transmittance, dlog_transmittance, i = arg
      splat_idx = work_splat_idx[i]
      cov = ((splats.cov[splat_idx, 0, 0], splats.cov[splat_idx, 0, 1]),
             (splats.cov[splat_idx, 1, 0], splats.cov[splat_idx, 1, 1]),)
      splat_xy = splats.xyd[splat_idx, 0], splats.xyd[splat_idx, 1]
      tile_xy = (tile.xy[:, :, 0], tile.xy[:, :, 1])
      splat_alpha = splats.alpha[splat_idx]

      # First, we have to recover the transmittance at the end of handling the
      # previous splat.
      def compute_prev_transmittance(log_transmittance):
        centered_xy = [a - b for a, b in zip(tile_xy, splat_xy)]
        solve_xy = _solve_2d(cov, centered_xy)
        dot_out = centered_xy[0] * solve_xy[0] + centered_xy[1] * solve_xy[1]
        density = jnp.exp(-.5 * dot_out)
        alpha = density * splat_alpha
        is_valid = (i < work_start + final_count)
        return jnp.float32(log_transmittance - jnp.log1p(-alpha * is_valid))

      log_transmittance = compute_prev_transmittance(log_transmittance)

      # Next, we'll autodiff the forward computation to compute gradients.
      def f_fwd(cov, splat_xy, tile_xy, alpha, log_transmittance, color):
        centered_xy = [a - b for a, b in zip(tile_xy, splat_xy)]
        solve_xy = _solve_2d(cov, centered_xy)
        dot_out = centered_xy[0] * solve_xy[0] + centered_xy[1] * solve_xy[1]
        density = jnp.exp(-.5 * dot_out)
        alpha = density * alpha
        is_valid = (i < work_start + final_count)
        incremental_color = (
            (is_valid * jnp.exp(log_transmittance) * alpha)[..., None] * color)
        log_transmittance += jnp.log1p(-alpha * is_valid)
        return jnp.float32(incremental_color), jnp.float32(log_transmittance)

      splat_color = splats.color[splat_idx, :]
      _, f_vjp = jax.vjp(
          f_fwd,
          cov, splat_xy, tile_xy, splat_alpha, log_transmittance, splat_color)
      dcov, dsplat_xy, dtile_xy, dalpha, dlog_transmittance, dsplat_color = (
          jtu.tree_map(jnp.float32, f_vjp((dfinal_color, dlog_transmittance))))

      ((dsplats.cov[i, 0, 0], dsplats.cov[i, 0, 1]),
       (dsplats.cov[i, 1, 0], dsplats.cov[i, 1, 1])) = dcov
      dsplats.alpha[i] = dalpha
      dsplats.color[i] = dsplat_color
      dsplats.xyd[i, 0], dsplats.xyd[i, 1] = dsplat_xy

      dxy = jtu.tree_map(lambda a, b: a + b, dxy, dtile_xy)
      return (dxy, jnp.float32(log_transmittance),
              jnp.float32(dlog_transmittance), i - 1)

    log_transmittance = jnp.float32(log_transmittance)
    dlog_transmittance = jnp.zeros_like(log_transmittance)
    dxy = (jnp.zeros(tile.xy.shape[:-1]),) * 2
    dxy, *_ = jax.lax.fori_loop(
        0, final_count.max(), body,
        (dxy, log_transmittance, dlog_transmittance,
         final_count.max() - 1 + work_start))
    dtile.xy[:, :, 0], dtile.xy[:, :, 1] = dxy

  splats_arg = splats._replace(radius=(),
                               color=jnp.pad(splats.color, ((0, 0), (0, 1))))

  dtiles, dwork_splats = _bwd_render_img_pallas(  # pylint:disable=unpacking-non-sequence,no-value-for-parameter
      jtu.tree_map(lambda t: jnp.zeros((workspace_dim, *t.shape[1:]), t.dtype),
                   splats_arg),  # initializer for dsplats
      Tile(xy=xys,
           work_start=tile_work_bounds[:-1].reshape(ny, nx),
           work_limit=tile_work_bounds[1:].reshape(ny, nx)),
      splats_arg,
      work_splat_idx,
      final_count,
      final_transmittance,
      jnp.pad(dfinal_color, ((0, 0), (0, 0), (0, 1))))

  locs = jnp.nonzero(dwork_splats.alpha, size=dwork_splats.alpha.shape[0])
  jax.debug.print(
      'dalpha={dalpha} at {locs}',
      locs=locs[:20],
      dalpha=dwork_splats.alpha[locs][:20])

  # Since dsplats is work-item-indexed, we will segment-sum.
  num_splats = splats.xyd.shape[0]
  dsplats = jtu.tree_map(
      # TODO: If needed for numerics, use bucket_size=..
      lambda t: jax.ops.segment_sum(t, work_splat_idx, num_segments=num_splats),
      dwork_splats)
  # TODO: Why can't we replace with radius=None?
  dsplats = dsplats._replace(radius=jnp.zeros_like(splats.radius),
                             color=dsplats.color[..., :3])

  # args: xys, splats, tile_work_bounds, work_splat_idx
  return (dtiles.xy, dsplats, None, None)


_render_img.defvjp(_render_img_fwd, _render_img_bwd)


@jax.jit
def _tile_for_splat_and_idx(splats, splat_idx, tile_idx_for_splat, grid):
  """Computes a tile index."""
  # For a given splat index and splat-tile-correspondence-index, computes the
  # image-space index of the tile. A splat may match 2x2 tiles, in which case
  # the correspondence indices will be 0, 1, 2, 3. If the splat's top left
  # tile is at (3, 4), then the tile indices will be, respectively, those for
  # tiles (3, 4), (3, 5), (4, 4), and (4, 5).
  (min_tx, min_ty), (max_tx, _) = util.get_tile_rect(
      splats.xyd[splat_idx, :2], splats.radius[splat_idx], grid)
  ncol = max_tx - min_tx
  offset_row = tile_idx_for_splat // ncol
  offset_col = tile_idx_for_splat % ncol
  return jnp.where(
      splat_idx < splats.xyd.shape[0],
      (min_ty + offset_row) * grid.nx + min_tx + offset_col,
      jnp.iinfo(jnp.int32).max)


def _make_x64_fn(fn):
  """Decorator that enables x64 in forward evaluation as well as AD."""
  @jax.custom_jvp
  def f(*args, **kwargs):
    x64_enabled = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
      return fn(*args, **kwargs)
    finally:
      jax.config.update('jax_enable_x64', x64_enabled)

  def f_jvp(primals, tangents):
    x64_enabled = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    try:
      return jax.jvp(f, primals, tangents)
    finally:
      jax.config.update('jax_enable_x64', x64_enabled)
  f.defjvp(f_jvp)

  return f


@jax.jit
@_make_x64_fn
def _sort_tile_splats(tile_for_splat_idx, splat_depth_u32, splat_idx):
  """Sorts work into chunks by tile index, then by splat depth."""
  assert jax.config.x64_enabled
  # TODO: For some reason JAX really wants this to be uint32 in lowering.
  u64_full_32s = jnp.full(tile_for_splat_idx.shape, 32).astype(jnp.uint64)
  sort_keys = (
      # TODO: would prefer raw '32', but get "'stablehlo.shift_left' op
      # requires compatible types for all operands and results:"
      (tile_for_splat_idx.astype(jnp.uint64) << u64_full_32s) |
      splat_depth_u32)
  assert sort_keys.dtype == jnp.uint64
  work_tile_idx, work_splat_idx = _make_x64_fn(gpu_sort.sort_pairs)(
      sort_keys, splat_idx.astype(jnp.uint32))
  # TODO: would prefer raw '32', but get "'stablehlo.shift_left' op
  # requires compatible types for all operands and results:"
  work_tile_idx = (work_tile_idx >> u64_full_32s).astype(jnp.uint32)
  work_splat_idx = work_splat_idx.astype(jnp.uint32)
  return work_tile_idx, work_splat_idx


@functools.partial(
    jax.jit,
    static_argnames=('grid', 'workspace_dim', 'use_pallas', 'color_clipping',
                     'interpret', 'debug'))
def render(xys: jax.Array,
           *,
           splats: Splat = None,
           blobs: Blob = None,
           camera: Camera = None,
           grid: Grid,
           workspace_dim: int = 10_000_000,
           use_pallas: bool = True,
           color_clipping: str = 'linear_clip',
           interpret: bool = False,
           debug: bool = False,
           ) -> tuple[jax.Array, Mapping[str, jax.Array]]:
  """Renders an image using 3d gaussian splatting."""
  if splats is None:
    def mk_splat(b):
      return splat_blob(b, camera, color_clipping=color_clipping)
    splats = jax.vmap(mk_splat)(blobs)
  splats = splats._replace(
      alpha=splats.alpha * util.splat_inlier(splats, grid.bounds))

  if not use_pallas:
    idx = jnp.argsort(splats.xyd[..., -1])
    sorted_splats = jtu.tree_map(lambda t: t[idx], splats)
    return (
        jax.vmap(jax.vmap(
            lambda xy: render_sorted_splats(xy, sorted_splats)))(xys),
        {})  # aux dict
  img_y, img_x = xys.shape[:2]
  ny, nx = (img_y + BLOCK_Y - 1) // BLOCK_Y, (img_x + BLOCK_X - 1) // BLOCK_X
  tile_grid = grid._replace(nx=nx, ny=ny)
  # How many tiles might be touched by each splat?
  splat_touched_tiles = jax.vmap(
      lambda spl: util.count_touched_tiles(spl.xyd[:2], spl.radius, tile_grid))(
          splats) * (splats.alpha > 0) * util.in_frustum(splats, grid.bounds)
  splat_touched_tiles = splat_touched_tiles.astype(jnp.uint32)
  # Cumsum of the above, which tells us split points for the workspace.
  cum_touched_tiles = jnp.cumsum(splat_touched_tiles)
  # Splat index (all the same splat_idx between split points), monotonic.
  splat_idx = jnp.searchsorted(
      cum_touched_tiles, jnp.arange(workspace_dim), side='right',
      method='scan_unrolled')
  tiles_before_this_splat = cum_touched_tiles[splat_idx]
  tiles_this_splat = splat_touched_tiles[splat_idx]
  # Ascending ranges from 0 to `tiles_this_splat - 1`.
  tile_for_this_splat = (
      (jnp.arange(workspace_dim) - tiles_before_this_splat) % tiles_this_splat)

  # Corresponding global tile indices (for sort key).
  tile_for_splat_idx = jax.vmap(
      _tile_for_splat_and_idx, in_axes=(None, 0, 0, None))(
          splats, splat_idx, tile_for_this_splat, tile_grid)
  # Sort by global tile index and by depth of the splat.
  # TODO: Sort by z?
  splat_depth_u32 = jnp.float32(splats.xyd[splat_idx, 2]).view(jnp.uint32)
  work_tile_idx, work_splat_idx = _sort_tile_splats(
      tile_for_splat_idx, splat_depth_u32, splat_idx)
  # The work is now sorted into sequences of splats for each tile to do work
  # against. `work_tile_idx`` tells us which tile, and `work_splat_idx`` tells
  # us which splat.

  tile_work_bounds = jnp.searchsorted(work_tile_idx, jnp.arange(nx * ny + 1),
                                      method='scan_unrolled')
  # `tile_work_bounds` gives us start and end points for each tile's work range.

  mean_rad_per_tile = (
      jax.ops.segment_sum(
          splats.radius[work_splat_idx], work_tile_idx, num_segments=nx * ny) /
      jnp.diff(tile_work_bounds))
  im = _render_img(FrozenDict(interpret=interpret, debug=debug),
                   xys, splats, tile_work_bounds, work_splat_idx)
  aux = dict(workspace_used=cum_touched_tiles[-1],
             workspace_size=workspace_dim,
             workspace_occupancy=cum_touched_tiles[-1] / workspace_dim,
             splat_touched_tiles=splat_touched_tiles,
             tile_work_bounds=tile_work_bounds,
             tile_median_radius=mean_rad_per_tile)
  if debug:
    _, extra = jax.lax.stop_gradient(_render_img_fwd(
        FrozenDict(interpret=interpret, debug=debug),
        *jax.lax.stop_gradient(
            (xys, splats, tile_work_bounds, work_splat_idx))))
    *_, final_count, final_transmittance = extra
    aux.update(dict(final_count=final_count,
                    final_transmittance=final_transmittance))
  return im, aux
