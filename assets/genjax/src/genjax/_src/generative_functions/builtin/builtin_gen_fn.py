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

from dataclasses import dataclass

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.pytree import DynamicClosure
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms.incremental import static_check_tree_leaves_diff
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_datatypes import BuiltinChoiceMap
from genjax._src.generative_functions.builtin.builtin_datatypes import BuiltinTrace
from genjax._src.generative_functions.builtin.builtin_primitives import _inline
from genjax._src.generative_functions.builtin.builtin_primitives import cache
from genjax._src.generative_functions.builtin.builtin_primitives import trace
from genjax._src.generative_functions.builtin.builtin_tracetype import (
    trace_type_transform,
)
from genjax._src.generative_functions.builtin.builtin_transforms import assess_transform
from genjax._src.generative_functions.builtin.builtin_transforms import (
    importance_transform,
)
from genjax._src.generative_functions.builtin.builtin_transforms import (
    simulate_transform,
)
from genjax._src.generative_functions.builtin.builtin_transforms import update_transform


#####
# Builtin language syntactic sugar
#####

# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the builtin language for functions via the `cache` builtin_primitives.
@dataclass
class DeferredFunctionCall(Pytree):
    fn: Callable
    kwargs: Dict
    args: Union[None, Tuple]

    def flatten(self):
        return (self.args,), (self.fn, self.kwargs)

    @classmethod
    def new(cls, fn, **kwargs):
        assert not isinstance(fn, GenerativeFunction)
        return DeferredFunctionCall(fn, kwargs, None)

    def __call__(self, *args):
        return DeferredFunctionCall(self.fn, self.kwargs, args)

    def __matmul__(self, addr):
        return cache(addr, self.fn, **self.kwargs)(*self.args)


def save(fn, **kwargs):
    return DeferredFunctionCall.new(fn, **kwargs)


# Denotes that a generative function should be inlined in the
# `@` syntactic sugar for addressing.
class INLINE_FLAG:
    pass


inline = INLINE_FLAG()

# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the builtin language for generative functions via the `trace` intrinsic.
@dataclass
class DeferredGenerativeFunctionCall(Pytree):
    gen_fn: GenerativeFunction
    kwargs: Dict
    args: Tuple

    def flatten(self):
        return (self.args,), (self.gen_fn, self.kwargs)

    @classmethod
    def new(cls, gen_fn, args, kwargs):
        return DeferredGenerativeFunctionCall(gen_fn, kwargs, args)

    def __matmul__(self, addr):
        if addr == inline:
            # To use inlining, the generative function must be a
            # `BuiltinGenerativeFunction`.
            assert isinstance(self.gen_fn, BuiltinGenerativeFunction)
            return _inline(self.gen_fn, *self.args, **self.kwargs)
        else:
            return trace(addr, self.gen_fn, **self.kwargs)(*self.args)


# This mixin overloads the call functionality for this generative function
# and allows usage of shorthand notation in the builtin DSL.
class SupportsBuiltinSugar:
    @dispatch
    def __call__(self, *args: Any, **kwargs) -> DeferredGenerativeFunctionCall:
        return DeferredGenerativeFunctionCall.new(self, args, kwargs)

    @dispatch
    def __call__(self, key: PRNGKey, args: Tuple) -> Any:
        key, tr = self.simulate(key, args)
        retval = tr.get_retval()
        return key, retval


#####
# Generative function
#####


@dataclass
class BuiltinGenerativeFunction(GenerativeFunction, SupportsBuiltinSugar):
    source: Callable

    def flatten(self):
        if isinstance(self.source, DynamicClosure):
            return (self.source,), ()
        else:
            return (), (self.source,)

    @typecheck
    @classmethod
    def new(cls, source: Callable):
        return BuiltinGenerativeFunction(source)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        return trace_type_transform(self.source, **kwargs)(*args)

    @typecheck
    def simulate(
        self, key: PRNGKey, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, BuiltinTrace]:
        key, (f, args, r, chm, score), cache = simulate_transform(
            self.source, **kwargs
        )(key, args)
        # `chm` is a `Trie` here.
        if not chm.inner:
            chm = EmptyChoiceMap()
        return key, BuiltinTrace(self, args, r, chm, cache, score)

    @typecheck
    def importance(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[FloatArray, BuiltinTrace]]:
        key, (w, (f, args, r, chm, score)), cache = importance_transform(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, cache, score))

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: ChoiceMap,
        argdiffs: Tuple,
        **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:
        assert static_check_tree_leaves_diff(argdiffs)
        (
            key,
            (
                retval_diffs,
                w,
                (f, args, r, chm, score),
                discard,
            ),
            cache,
        ) = update_transform(self.source, **kwargs)(key, prev, constraints, argdiffs)
        return key, (
            retval_diffs,
            w,
            BuiltinTrace(self, args, r, chm, cache, score),
            BuiltinChoiceMap(discard),
        )

    @typecheck
    def assess(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        key, (retval, score) = assess_transform(self.source, **kwargs)(key, chm, args)
        return key, (retval, score)

    def inline(self, *args):
        return _inline(self, *args)


#####
# Partial binding / currying
#####


def partial(gen_fn, *static_args):
    return BuiltinGenerativeFunction.new(
        lambda *args: gen_fn.inline(*args, *static_args),
    )


##############
# Shorthands #
##############

lang = BuiltinGenerativeFunction.new
