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
"""This module contains definitions for interpreters which act as program
transformers when staged out by JAX.

These interpreters support different patterns of program transformation.
Mostly, each implementation contains similar functionality, but are kept
separate to allow customization (here, I mean things like interpretation
environments, or abstract types which define the object-level values the
interpreter produces for each primitive statement in the Jaxpr).
"""
