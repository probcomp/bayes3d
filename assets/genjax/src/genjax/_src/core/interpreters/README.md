# GenJAX core interpreters

This directory contains several interpreters. These interpreters form the basis layer for functionality which relies on program transformations to implement semantics.

Several of these interpreters have been collected and modified from notebooks or other JAX-based systems. For each interpreter / transformation implementation, we've tried to keep a reference to the original attribution.

This README contains a brief description of each interpreter.

## Context

**Implementation:** `context.py`

A final style (customized trace/tracer) interpreter (it defines its own `Trace` and `Tracers`) which wraps `Jaxpr` interpreters (defined in the module: `Fwd` and `Cont`). Allows dispatching registered primitives to a dynamic context (which the user can inherit and override). The context can store and yield state - allowing lifting of pure functions to ones which accept and return state.

### Supports

* (**GFI implementations for `Builtin` language**)
* (**harvest**) Oryx's `harvest` transformation.
* (**diff_rules**) Forward metadata propagation transformation.
* (**adev**) [ADEV](https://arxiv.org/pdf/2212.06386.pdf) CPS transform with dual number propagation.

## Propagate

**Implementation:** `propagate.py`

[Original from Oryx][oryx_propagate]

An initial style interpreter (no special `Trace` or `Tracers`) which treats a `Jaxpr` as a graph, primitives as edges, and variables as nodes - and attempts to compute using fixpoint iteration on a lifted type lattice (short: allows forward + backward propagation of abstract values).

### Supports

* (**`coryx`**) A DSL for exact densities based on Oryx.

[effects_notebook]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[oryx_propagate]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/propagate.py
[oryx_harvest]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/harvest.py
