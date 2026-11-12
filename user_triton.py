"""Replay user-defined Triton kernels in a saved full graph.

A saved graph calls each user kernel through ``triton_kernel_wrapper_functional``
with a ``kernel_idx`` into the capturing process's ``kernel_side_table``, which
no other process holds. A graph is replayable only when its sidecar carries a
kernel bundle (``user_triton_bundle``) written at ingest, and a resolver
registered in this process turns every bundle entry into a kernel object
imported from reviewed source.

Then the loader rewrites each HOP call site in the graph class into a call to
``launch_user_triton``, which runs the kernel through ``wrap_triton``. Under
torch.compile, Dynamo traces that like the original model's own
``wrap_triton`` launch and registers the kernel at trace time, so the graph
never depends on the side table: not on the captured index, and not on what
``torch._dynamo.reset()`` leaves behind. Eagerly it launches the kernel
directly.

Nothing here executes kernel source taken from a trace: the rewrite only
changes call sites, and the kernel objects come from the resolver.

The resolver is ``MODULE:CALLABLE`` in the ``BETTER_BENCHMARK_USER_TRITON_RESOLVER``
environment variable, so bench worker subprocesses inherit it. It is called
with one bundle entry and returns the kernel to launch, or raises
``UserTritonResolutionError``.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
from typing import Any, Callable, NamedTuple

BUNDLE_KEY = "user_triton_bundle"
BUNDLE_SCHEMA_VERSION = 1
RESOLVER_ENV = "BETTER_BENCHMARK_USER_TRITON_RESOLVER"
LAUNCHER_NAME = "launch_user_triton"
FUNCTIONAL_HOP = "torch.ops.higher_order.triton_kernel_wrapper_functional"
# HOP keywords the launcher takes; a call with any other (e.g. tensor_bases)
# is not rewritten.
_LAUNCHER_KEYWORDS = frozenset({
    "kernel_idx", "constant_args_idx", "grid", "tma_descriptor_metadata",
    "kwargs", "tensors_to_clone", "launch_kwargs",
})

# Kernel names launched with one pipeline stage because their captured configs
# ran out of shared memory (see make_launcher); read by the numerics gate.
SINGLE_STAGE_RETRIES: list[str] = []

_resolved: dict[tuple[str, str], Any] = {}


class UserTritonResolutionError(RuntimeError):
    """A bundle entry cannot be turned into a trusted kernel object."""


class ResolvedBundle(NamedTuple):
    # captured kernel_idx -> kernel object; captured constant_args_idx -> args
    kernels: dict[int, Any]
    constant_args: dict[int, dict[str, Any]]


def _resolver() -> Callable[[dict[str, Any]], Any] | None:
    spec = os.environ.get(RESOLVER_ENV)
    if not spec:
        return None
    module, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module), attr)


def resolve_bundle(
    sidecar: dict[str, Any] | None, *, scope: str | None = None
) -> ResolvedBundle | None:
    """Kernel object per captured kernel_idx for every entry of the sidecar's
    bundle, plus the captured constant args; None when the graph has no
    bundle, no resolver is registered, or any entry fails to resolve (then the
    graph stays unreplayable as a whole).

    Kernel objects are cached per ``scope`` (the graph's path), so reloading a
    graph reuses them; with no scope nothing is cached. They are never shared
    across graphs: a resolved autotuner may cache its tuned config with a key
    that does not see the graph's shapes."""
    bundle = (sidecar or {}).get(BUNDLE_KEY)
    if not bundle:
        return None
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        print(f"[user_triton] unsupported bundle schema {bundle.get('schema_version')!r}",
              file=sys.stderr)
        return None
    resolver = _resolver()
    if resolver is None:
        print(f"[user_triton] graph has a kernel bundle but {RESOLVER_ENV} is unset",
              file=sys.stderr)
        return None
    kernels = {}
    for idx, entry in bundle.get("kernels", {}).items():
        # One kernel object per distinct entry (kid + configs) per graph, so
        # repeated loads of a graph do not grow Triton's caches.
        key = (scope, json.dumps(entry, sort_keys=True))
        try:
            kernel = _resolved.get(key) if scope is not None else None
            if kernel is None:
                kernel = resolver(entry)
                if scope is not None:
                    _resolved[key] = kernel
        except Exception as error:  # noqa: BLE001 - any failure leaves the graph refused
            print(f"[user_triton] kernel_idx {idx} ({entry.get('name')}): {error}",
                  file=sys.stderr)
            return None
        kernels[int(idx)] = kernel
    constant_args = {int(idx): args for idx, args in bundle.get("constant_args", {}).items()}
    return ResolvedBundle(kernels, constant_args)


def hop_call_indices(graph_class: ast.ClassDef) -> list[tuple[int | None, int | None]]:
    """(kernel_idx, constant_args_idx) of every functional HOP call in the
    class; None where the argument is not an int literal."""

    def literal(node: ast.Call, name: str) -> int | None:
        value = next((k.value for k in node.keywords if k.arg == name), None)
        if isinstance(value, ast.Constant) and isinstance(value.value, int):
            return value.value
        return None

    return [
        (literal(node, "kernel_idx"), literal(node, "constant_args_idx"))
        for node in ast.walk(graph_class)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == FUNCTIONAL_HOP
    ]


def rewrite_hop_calls(graph_class: ast.ClassDef, bundle: ResolvedBundle) -> bool:
    """Point every functional HOP call at the launcher, in place. Rewrites
    nothing unless every call's kernel and constant args are in the bundle,
    and never touches mutation HOPs (they stay and keep the graph refused)."""
    if any(isinstance(n, ast.Name) and n.id == LAUNCHER_NAME for n in ast.walk(graph_class)):
        return False
    calls = hop_call_indices(graph_class)
    if not calls or any(
        kidx not in bundle.kernels or cidx not in bundle.constant_args for kidx, cidx in calls
    ):
        return False
    nodes = [
        node
        for node in ast.walk(graph_class)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == FUNCTIONAL_HOP
    ]
    keywords = [{k.arg: k.value for k in node.keywords} for node in nodes]
    # Host-side TMA descriptors and tensor_bases are not replayed.
    if any(
        node.args
        or set(kw) - _LAUNCHER_KEYWORDS
        or not (isinstance(kw.get("tma_descriptor_metadata"), ast.Dict)
                and not kw["tma_descriptor_metadata"].keys)
        for node, kw in zip(nodes, keywords)
    ):
        return False
    grids = [kw.get("grid") for kw in keywords]
    # The HOP holds one grid per autotune config. Only calls whose configs all
    # share one grid are replayed, so the launcher passes wrap_triton that grid.
    if any(
        not isinstance(grid, ast.List)
        or not grid.elts
        or any(ast.dump(g) != ast.dump(grid.elts[0]) for g in grid.elts)
        for grid in grids
    ):
        return False
    for node, grid in zip(nodes, grids):
        node.func = ast.copy_location(ast.Name(id=LAUNCHER_NAME, ctx=ast.Load()), node.func)
        grid.elts = grid.elts[:1]
    return True


def launches_user_triton(module: Any) -> bool:
    """Whether a loaded graph module's forward calls rewritten user kernels."""
    forward = getattr(type(module), "forward", None)
    return LAUNCHER_NAME in getattr(getattr(forward, "__code__", None), "co_names", ())


def make_launcher(bundle: ResolvedBundle) -> Callable[..., dict[str, Any]]:
    """Drop-in for triton_kernel_wrapper_functional at a rewritten call site."""
    import torch
    from torch._library.triton import wrap_triton

    def _tracing(args: dict[str, Any]) -> bool:
        from torch._subclasses.fake_tensor import is_fake
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        return get_proxy_mode() is not None or any(
            isinstance(val, torch.Tensor) and is_fake(val) for val in args.values()
        )

    def _clone(val: torch.Tensor) -> torch.Tensor:
        # Strides kept, as the kernel addresses the clone with the captured
        # stride arguments; written so Dynamo can trace it.
        out = torch.empty_strided(val.size(), val.stride(), dtype=val.dtype, device=val.device)
        return out.copy_(val)

    def _alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        return torch.empty(size, device="cuda", dtype=torch.int8)

    single_stage: dict[int, Any] = {}

    def _launch_direct(kernel_idx: int, grid: Any, args: dict[str, Any],
                       constant_args: dict[str, Any]) -> None:
        import copy

        import triton
        from triton.runtime.errors import OutOfResources

        kernel = bundle.kernels[kernel_idx]
        triton.set_allocator(_alloc)
        try:
            kernel[grid](**args, **constant_args)
            return
        except OutOfResources:
            # Configs captured for low-precision inputs can exceed shared
            # memory when the upcast numerics reference runs the kernel in
            # float32. Retry with one pipeline stage: same tiles and grid,
            # same math, only less prefetching.
            if not getattr(kernel, "configs", None):
                raise
        SINGLE_STAGE_RETRIES.append(getattr(kernel.fn, "__name__", str(kernel_idx)))
        if kernel_idx not in single_stage:
            configs = []
            for config in kernel.configs:
                config = copy.copy(config)
                config.num_stages = 1
                configs.append(config)
            single_stage[kernel_idx] = triton.autotune(configs=configs, key=[])(kernel.fn)
        single_stage[kernel_idx][grid](**args, **constant_args)

    def launch_user_triton(*, kernel_idx, constant_args_idx, grid,
                           tma_descriptor_metadata, kwargs, tensors_to_clone,
                           launch_kwargs=None):
        kernel = bundle.kernels[kernel_idx]
        # Same clone-then-mutate semantics as the functional HOP.
        args = {
            key: _clone(val) if key in tensors_to_clone else val
            for key, val in kwargs.items()
        }
        constant_args = bundle.constant_args[constant_args_idx]
        # No builtins in here: guarding a builtin read here fails in Dynamo
        # (KeyError) for graph classes, which run with restricted builtins.
        if torch.compiler.is_compiling() or _tracing(args):
            wrap_triton(kernel)[grid[0]](**args, **constant_args)
        else:
            # Real tensors: launch directly, also under numerics dispatch modes
            # (the upcast reference), where wrap_triton would emit a HOP the
            # mode has no rule for. Kernels that build TMA descriptors on
            # device need a scratch allocator; the model's own wrapper set one,
            # and Inductor sets one for compiled code.
            _launch_direct(kernel_idx, grid[0], args, constant_args)
        return {key: val for key, val in args.items() if key in tensors_to_clone}

    return launch_user_triton


WRITE_ANALYSIS_KEY = "better_benchmark_user_triton_write_analysis"


class TrustedKernels:
    """Kernel objects for captured kernel indices whose identity a loader
    checked against reviewed source, with the captured constant args.

    Stored in ``graph_module.meta[WRITE_ANALYSIS_KEY]`` so the partitioner
    can ask which tensors a launch writes (``written_kwargs``) instead of
    assuming it writes every tensor it takes. Shared, not copied, when the
    graph module is deep-copied.
    """

    def __init__(self, kernels: dict[int, Any], constant_args: dict[int, dict[str, Any]]):
        self.kernels = kernels
        self.constant_args = constant_args
        self._writes: dict[tuple, frozenset[str] | None] = {}

    def __deepcopy__(self, memo: dict[int, Any]) -> "TrustedKernels":
        return self

    def written_kwargs(self, node: Any) -> frozenset[str] | None:
        """Names of the tensor kwargs a ``triton_kernel_wrapper_mutation``
        launch writes, from Triton's TTIR mutation analysis; None when the
        kernel or its constant args are not trusted, or a value is unknown."""
        import torch

        kernel_idx = node.kwargs.get("kernel_idx")
        constant_args_idx = node.kwargs.get("constant_args_idx")
        if kernel_idx not in self.kernels or constant_args_idx not in self.constant_args:
            return None
        values = {}
        for name, arg in (node.kwargs.get("kwargs") or {}).items():
            value = arg.meta.get("val") if hasattr(arg, "meta") else arg
            if isinstance(value, torch.SymInt):
                value = value.node.hint if not value.node.expr.is_number else int(value.node.expr)
                if value is None:
                    return None
            if value is None and hasattr(arg, "meta"):
                return None
            values[name] = value
        tensors = frozenset(n for n, v in values.items() if isinstance(v, torch.Tensor))
        key = (kernel_idx, constant_args_idx, tensors)
        if key not in self._writes:
            self._writes[key] = self._analyze(
                self.kernels[kernel_idx],
                {**values, **self.constant_args[constant_args_idx]},
                node.kwargs.get("tma_descriptor_metadata") or {},
                tensors,
            )
        return self._writes[key]

    @staticmethod
    def _analyze(kernel, kwargs, tma_descriptor_metadata, tensors) -> frozenset[str] | None:
        # A failed analysis reports every tensor as written.
        from torch._higher_order_ops.triton_kernel_wrap import identify_accessed_tensors

        try:
            accesses = identify_accessed_tensors(kernel, kwargs, tma_descriptor_metadata)
        except Exception:  # noqa: BLE001 - fall back to "writes every tensor"
            return None
        return frozenset(dep.name for dep in accesses.read_writes.writes if dep.name in tensors)
