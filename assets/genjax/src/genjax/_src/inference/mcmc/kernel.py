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

import abc
from dataclasses import dataclass

from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import PRNGKey


@dataclass
class MCMCKernel(Pytree):
    @abc.abstractmethod
    def reversal(self):
        pass

    @abc.abstractmethod
    def apply(self, key: PRNGKey, trace: Trace, *args):
        pass

    def __call__(self, key, trace, *args):
        return self.apply(key, trace, *args)
