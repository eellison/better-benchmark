"""Runtime for custom-kernel repros.

A custom-kernel repro (meta.json ``"kind": "custom_kernel"``) is one launch
of a kernel Inductor does not generate, at the shapes a captured graph
launched it with: a custom operator (``kernel_op``) or a user-defined Triton
kernel (``user_triton_launcher``). repro_harness times it with
``--inputs``/``--length-arm``/``--index-arm``/``--sensitivity``.

Nothing here executes kernel source taken from a trace. A user-Triton kernel
comes from the resolver registered in ``user_triton.RESOLVER_ENV``, which
checks the captured kernel identity against reviewed source. A custom
operator is called through ``torch.ops`` unless the hook in ``STANDINS_ENV``
(``MODULE:CALLABLE``, called with the operator's qualified name) returns a
replacement: a stand-in implementation for hardware the operator does not
support, or one whose ``prepare`` builds inputs the launch reads, such as the
forward outputs a backward kernel takes.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

STANDINS_ENV = "BETTER_BENCHMARK_KERNEL_STANDINS"


class TorchOp:
    """A custom operator called as captured."""

    note = "as captured"

    def __init__(self, qualname: str) -> None:
        import torch

        namespace, name = qualname.split("::")
        self.qualname = qualname
        self.op = getattr(getattr(torch.ops, namespace), name)

    def prepare(self, args: tuple) -> tuple:
        return args

    def __call__(self, *args: Any) -> Any:
        return self.op(*args)


def kernel_op(qualname: str) -> Any:
    """The callable a repro launches for custom operator ``qualname``: the
    ``STANDINS_ENV`` hook's replacement, or the operator itself."""
    spec = os.environ.get(STANDINS_ENV)
    if spec:
        module, _, attr = spec.partition(":")
        replacement = getattr(importlib.import_module(module), attr)(qualname)
        if replacement is not None:
            return replacement
    return TorchOp(qualname)


def user_triton_launcher(
    entry: dict[str, Any], grids: list[list[int]], constant_args: dict[str, Any]
) -> Callable[..., None]:
    """Launch function of one user-Triton kernel: the resolved kernel with
    its captured configs, launched with the captured grid of whichever
    config the autotuner picks."""
    import torch
    import triton
    from user_triton import _resolver, UserTritonResolutionError

    resolver = _resolver()
    if resolver is None:
        raise UserTritonResolutionError("no user-Triton resolver is registered")
    kernel = resolver(entry)
    configs = entry.get("configs") or []
    grids = [tuple(g) for g in grids]

    def grid(meta: dict[str, Any]) -> tuple:
        if len(set(grids)) == 1 or not configs:
            return grids[0]
        for config, g in zip(configs, grids):
            if all(meta.get(k) == v for k, v in config["kwargs"].items()):
                return g
        raise RuntimeError(f"no captured grid for config {meta}")

    def alloc(size: int, alignment: int, stream: int | None) -> torch.Tensor:
        return torch.empty(size, device="cuda", dtype=torch.int8)

    def launch(**kwargs: Any) -> None:
        triton.set_allocator(alloc)
        kernel[grid](**kwargs, **constant_args)

    return launch
