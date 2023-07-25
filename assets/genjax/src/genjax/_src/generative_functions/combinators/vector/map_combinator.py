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
"""This module implements a generative function combinator which allows
broadcasting for generative functions -- mapping over vectorial versions of
their arguments."""

from dataclasses import dataclass

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.masks import mask
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import concrete_cond
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import SupportsBuiltinSugar
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorTrace,
)
from genjax._src.generative_functions.combinators.vector.vector_tracetypes import (
    VectorTraceType,
)
from genjax._src.utilities import slash


#####
# MapCombinator
#####


@dataclass
class MapCombinator(GenerativeFunction, SupportsBuiltinSugar):
    """> `MapCombinator` accepts a generative function as input and provides
    `vmap`-based implementations of the generative function interface methods.

    Examples:

        ```python exec="yes" source="tabbed-left"
        import jax
        import jax.numpy as jnp
        import genjax
        console = genjax.pretty()

        @genjax.gen
        def add_normal_noise(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        # Creating a `MapCombinator` via the preferred `new` class method.
        mapped = genjax.MapCombinator.new(add_normal_noise, in_axes=(0,))

        key = jax.random.PRNGKey(314159)
        arr = jnp.ones(100)
        key, tr = jax.jit(genjax.simulate(mapped))(key, (arr, ))

        print(console.render(tr))
        ```
    """

    in_axes: Tuple
    repeats: Union[None, IntArray]
    kernel: GenerativeFunction

    def flatten(self):
        return (self.kernel,), (self.in_axes, self.repeats)

    @typecheck
    @classmethod
    def new(
        cls,
        kernel: GenerativeFunction,
        in_axes: Union[None, Tuple] = None,
        repeats: Union[None, IntArray] = None,
    ) -> "MapCombinator":
        """The preferred constructor for `MapCombinator` generative function
        instances. The shorthand symbol is `Map = MapCombinator.new`.

        Arguments:
            kernel: A single `GenerativeFunction` instance.
            in_axes: A tuple specifying which `args` to broadcast over.
            repeats: An integer specifying the length of repetitions (ignored if `in_axes` is specified, if `in_axes` is not specified - required).

        Returns:
            instance: A `MapCombinator` instance.
        """
        assert isinstance(kernel, GenerativeFunction)
        if in_axes is None or all(map(lambda v: v is None, in_axes)):
            assert repeats is not None
        return MapCombinator(in_axes, repeats, kernel)

    def _static_broadcast_dim_length(self, args):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(find_axis_size, self.in_axes, args)
        axis_sizes = set(jax.tree_util.tree_leaves(axis_sizes))
        if self.repeats is None and len(axis_sizes) == 1:
            (d_axis_size,) = axis_sizes
        elif len(axis_sizes) > 1:
            raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
        elif self.repeats is None:
            raise ValueError("repeats should be specified manually.")
        else:
            d_axis_size = self.repeats
        return d_axis_size

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        kernel_tt = self.kernel.get_trace_type(*args)
        return VectorTraceType(kernel_tt, broadcast_dim_length)

    @typecheck
    def simulate(
        self, key: PRNGKey, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, VectorTrace]:
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        assert len(args) == len(self.in_axes)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        indices = np.array([i for i in range(0, broadcast_dim_length)])
        key, sub_keys = slash(key, broadcast_dim_length)
        _, tr = jax.vmap(self.kernel.simulate, in_axes=(0, self.in_axes))(
            sub_keys, args
        )
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(scores))
        return key, map_tr

    def _bounds_checker(self, v, key_len):
        lengths = []

        def _inner(v):
            if v.shape[-1] > key_len:
                raise Exception("Length of leaf longer than max length.")
            else:
                lengths.append(v.shape[-1])
                return v

        ret = jtu.tree_map(_inner, v)
        fixed_len = set(lengths)
        assert len(fixed_len) == 1
        return ret, fixed_len.pop()

    def _static_check_trie_index_compatible(self, chm: Trie, broadcast_dim_length: Int):
        for (k, _) in chm.get_subtrees_shallow():
            assert isinstance(k, int)
            # TODO: pull outside loop, just check the last address.
            assert k < broadcast_dim_length

    def _importance_vcm(self, key, chm, args):
        def _importance(key, chm, args):
            return self.kernel.importance(key, chm, args)

        def _simulate(key, _, args):
            key, tr = self.kernel.simulate(key, args)
            return key, (0.0, tr)

        def _inner(key, index, chm, args):
            check = index == chm.get_index()
            return concrete_cond(check, _importance, _simulate, key, chm, args)

        broadcast_dim_length = self._static_broadcast_dim_length(args)
        indices = np.array([i for i in range(0, broadcast_dim_length)])
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (w, tr) = jax.vmap(_inner, in_axes=(0, 0, 0, self.in_axes))(
            sub_keys, indices, chm, args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VectorTrace(self, indices, tr, args, retval, scores)
        return key, (w, map_tr)

    # Implements a conversion from `Trie`.
    def _importance_tchm(self, key, chm, args):
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        self._static_check_trie_index_compatible(chm, broadcast_dim_length)

        # Okay, so the Trie has an address hierarchy which is compatible with the index structure of the MapCombinator choices.
        # Let's coerce Trie into VectorChoiceMap and then just call `_importance_vcm`.
        vector_chm = self._coerce_to_vector_chm(chm)
        return self._importance_vcm(key, vector_chm, args)

    def _importance_empty(self, key, _, args):
        key, map_tr = self.simulate(key, args)
        w = 0.0
        return key, (w, map_tr)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        chm: Union[EmptyChoiceMap, Trie, VectorChoiceMap],
        args: Tuple,
        **_,
    ) -> Tuple[PRNGKey, Tuple[FloatArray, VectorTrace]]:
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        assert len(args) == len(self.in_axes)
        # Note: these branches are resolved at tracing time.
        if isinstance(chm, VectorChoiceMap):
            return self._importance_vcm(key, chm, args)
        elif isinstance(chm, Trie):
            return self._importance_tchm(key, chm, args)
        else:
            assert isinstance(chm, EmptyChoiceMap)
            return self._importance_empty(key, chm, args)

    # The choice map passed in here is a vector choice map.
    def _update_vcm(
        self, key: PRNGKey, prev: VectorTrace, chm: VectorChoiceMap, argdiffs: Tuple
    ):
        def _update(key, prev, chm, argdiffs):
            key, (retdiff, w, tr, d) = self.kernel.update(key, prev, chm, argdiffs)
            return key, (retdiff, w, tr, d)

        def _inner(key, index, prev, chm, argdiffs):
            check = index == chm.get_index()
            masked = mask(check, chm.inner)
            return _update(key, prev, masked, argdiffs)

        # Just to determine the broadcast length.
        args = jtu.tree_leaves(argdiffs)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        indices = np.array([i for i in range(0, broadcast_dim_length)])
        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0, prev.inner
        )
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (retdiff, w, tr, discard) = jax.vmap(
            _inner, in_axes=(0, 0, prev_inaxes_tree, 0, self.in_axes)
        )(sub_keys, indices, prev.inner, chm, argdiffs)
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(scores))
        return key, (retdiff, w, map_tr, discard)

    # The choice map passed in here is empty, but perhaps
    # the arguments have changed.
    def _update_empty(self, key, prev, chm, argdiffs):
        def _fallback(key, prev, chm, argdiffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, EmptyChoiceMap(), argdiffs
            )
            return key, (retdiff, w, tr, d)

        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0, prev.inner
        )
        # Just to determine the broadcast length.
        args = jtu.tree_leaves(argdiffs)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (retdiff, w, tr, discard) = jax.vmap(
            _fallback, in_axes=(0, prev_inaxes_tree, 0, self.in_axes)
        )(sub_keys, prev.inner, chm, argdiffs)
        w = jnp.sum(w)
        indices = jnp.array([i for i in range(0, broadcast_dim_length)])
        map_tr = VectorTrace(self, indices, tr, jnp.sum(tr.get_score()))
        return key, (retdiff, w, map_tr, discard)

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: Union[EmptyChoiceMap, VectorChoiceMap],
        argdiffs: Tuple,
        **_,
    ):
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        assert len(argdiffs) == len(self.in_axes)
        # Branches here implement certain optimizations when more
        # information about the passed in choice map is available.
        if isinstance(chm, VectorChoiceMap):
            return self._update_vcm(key, prev, chm, argdiffs)
        else:
            assert isinstance(chm, EmptyChoiceMap)
            return self._update_empty(key, prev, chm, argdiffs)

    # TODO: I've had so many issues with getting this to work correctly
    # and not throw - and I'm not sure why it's been so finicky.
    # Investigate if it occurs again.
    def _throw_index_check_host_exception(
        self, check, truth: IntArray, index: IntArray
    ):
        def _inner(args, _):
            truth = args[0]
            index = args[1]
            check = args[2]
            if not np.all(check):
                raise Exception(
                    f"\nMapCombinator {self} received a choice map with mismatched indices in assess.\nReference:\n{truth}\nPassed in:\n{index}"
                )

        hcb.id_tap(
            _inner,
            (truth, index, check),
            result=None,
        )
        return None

    @typecheck
    def assess(
        self, key: PRNGKey, chm: VectorChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        assert len(args) == len(self.in_axes)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        indices = jnp.array([i for i in range(0, broadcast_dim_length)])
        check = jnp.count_nonzero(indices - chm.get_index()) == 0

        # This inserts a host callback check for bounds checking.
        # If there is an index failure, `assess` must fail
        # because we must provide a constraint for every generative
        # function call.
        self._throw_index_check_host_exception(check, indices, chm.get_index())

        inner = chm.inner
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (retval, score) = jax.vmap(self.kernel.assess, in_axes=(0, 0, self.in_axes))(
            sub_keys, inner, args
        )
        return key, (retval, jnp.sum(score))


##############
# Shorthands #
##############

Map = MapCombinator.new
