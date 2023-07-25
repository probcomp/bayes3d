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

import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Int
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.gensp.gensp_distribution import (
    GenSPDistribution,
)
from genjax._src.generative_functions.distributions.gensp.target import Target


@dataclasses.dataclass
class EntropyEstimatorsViaInference(Pytree):
    n_lower_bound: Int
    n_upper_bound: Int
    model: GenerativeFunction
    proposal: GenSPDistribution
    targets: Selection

    def flatten(self):
        return (self.model, self.proposal, self.targets), (
            self.n_lower_bound,
            self.n_upper_bound,
        )

    @typecheck
    @classmethod
    def new(
        cls,
        model: GenerativeFunction,
        proposal: GenSPDistribution,
        targets: Selection,
        n_lower_bound: Int,
        n_upper_bound: Int,
    ):
        return EntropyEstimatorsViaInference(
            n_lower_bound, n_upper_bound, model, proposal, targets
        )

    def _entropy_lower_bound(self, key: PRNGKey, model_args: Tuple):
        key, tr = self.model.simulate(key, model_args)
        obs_targets = self.targets.complement()
        observations = obs_targets.filter(tr.get_choices().strip())
        target = Target(self.model, model_args, observations)
        key, *sub_keys = jax.random.split(key, self.n_lower_bound + 1)
        sub_keys = jnp.array(sub_keys)
        _, tr_q = jax.vmap(self.proposal.simulate, in_axes=(0, None))(
            sub_keys, (target,)
        )
        (chm,) = tr_q.get_retval()
        chm_axes = jtu.tree_map(lambda v: 0, chm)
        observations_axes = jtu.tree_map(lambda v: None, observations)
        choices = observations.merge(chm)
        choices_axes = observations_axes.merge(chm_axes)
        key, *sub_keys = jax.random.split(key, self.n_lower_bound + 1)
        sub_keys = jnp.array(sub_keys)
        _, (log_p, _) = jax.vmap(
            self.model.importance, in_axes=(0, choices_axes, None)
        )(sub_keys, choices, model_args)
        log_q = tr_q.get_score()
        log_w = log_p - log_q
        return key, jnp.mean(log_w), (log_p, log_q)

    def _entropy_upper_bound(self, key: PRNGKey, model_args: Tuple):
        key, tr = self.model.simulate(key, model_args)
        log_p = tr.get_score()
        chm = tr.get_choices().strip()
        latents = self.targets.filter(chm)
        observations = self.targets.complement().filter(chm)
        target = Target(self.model, model_args, observations)
        key, *sub_keys = jax.random.split(key, self.n_upper_bound + 1)
        sub_keys = jnp.array(sub_keys)
        _, (log_q, _) = jax.vmap(self.proposal.assess, in_axes=(0, None, None))(
            sub_keys, ValueChoiceMap(latents), (target,)
        )
        log_w = log_q - log_p
        return key, -jnp.mean(log_w), (log_p, log_q)

    def estimate(self, key: PRNGKey, model_args: Tuple):
        _, lower_bound, (_, _) = self._entropy_lower_bound(
            key,
            model_args,
        )
        _, upper_bound, (_, _) = self._entropy_upper_bound(
            key,
            model_args,
        )
        return key, (-upper_bound, -lower_bound)

    def __call__(self, key: PRNGKey, model_args: Tuple):
        return self.estimate(key, model_args)


##############
# Shorthands #
##############

eevi = EntropyEstimatorsViaInference.new
