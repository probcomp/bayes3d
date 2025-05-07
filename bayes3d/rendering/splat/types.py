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
"""Types and aliases for splatting code."""

from typing import NamedTuple

import jax


Key = jax.Array
Array = jax.Array


# TODO: better covariance parameterization
class Blob(NamedTuple):
  """A 3D Gaussian with color and alpha."""
  # camera coordinates
  xyz: Array
  cov: Array  # [3, 3]
  color: Array  # Spherical harmonics coefficients
  alpha: Array


class Splat(NamedTuple):
  """A 2D splat of a Blob."""
  # image plane coordinates
  xyd: Array
  cov: Array  # [2, 2]
  radius: Array  # Max eigenvalue
  color: Array
  alpha: Array


class Camera(NamedTuple):
  xyz: Array  # 3D world coordinate of camera
  orientation: Array  # 3D world to camera projection matrix
  fx: float
  fy: float
  width: float
  height: float


class Bounds(NamedTuple):
  """A 2D bounding box."""
  minx: float
  maxx: float
  miny: float
  maxy: float


class Grid(NamedTuple):
  """A 2D Grid."""
  bounds: Bounds
  nx: int
  ny: int
