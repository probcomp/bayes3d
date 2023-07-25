# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module provides a function-like modeling language. The generative
function interfaces are implemented for objects in this language using
transformations by JAX interpreters.

The language also exposes a set of JAX primitives which allow
hierarchical construction of generative programs. These programs can
utilize other generative functions inside of a new JAX primitive
(`trace`) to create hierarchical patterns of generative computation.
"""
