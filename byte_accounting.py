"""Effective byte accounting for SOL benchmark baselines.

For a fused Inductor region, the useful bandwidth baseline is usually original
input reads plus final output writes. Sparse access ops are the exception: an
embedding or gather should not charge the whole source table when the kernel
only touches indexed rows/elements.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def tensor_alias_key(tensor: torch.Tensor) -> tuple[int, int, int, int]:
    try:
        storage_ptr = tensor.untyped_storage().data_ptr()
    except RuntimeError:
        storage_ptr = tensor.data_ptr()
    return (
        storage_ptr,
        tensor.storage_offset(),
        tensor.nelement(),
        tensor.element_size(),
    )


def count_unique_tensor_bytes(values: Any) -> int:
    """Count tensor bytes in a nested output, de-duping exact tensor aliases."""
    seen = set()
    total = 0
    for value in _iter_tensors(values):
        key = tensor_alias_key(value)
        if key in seen:
            continue
        seen.add(key)
        total += tensor_nbytes(value)
    return total


def count_bytes_naive(inputs: Any, outputs: Any) -> int:
    """Count all tensor input bytes plus unique output bytes."""
    return sum(tensor_nbytes(t) for t in _iter_tensors(inputs)) + count_unique_tensor_bytes(outputs)


def count_bytes_effective(mod, inputs: Iterable[Any]) -> int:
    """Count effective input reads plus final output writes for a repro module.

    Dense ops charge full original tensor inputs. Sparse source reads from
    embedding/gather/index/index_select charge the produced result size instead
    of the full source tensor. Sparse updates such as index_put/scatter charge
    final output writes by updated element count when the sparse-update result is
    returned from the region.
    """
    with _EffectiveByteCounter() as counter:
        with torch.no_grad():
            outputs = mod(*inputs)
    return counter.total(outputs)


def _iter_tensors(values: Any):
    for value in _pytree.tree_leaves(values):
        if isinstance(value, torch.Tensor):
            yield value


def _op_name(func) -> str:
    return str(func)


def _tensor_bytes(value: Any) -> int:
    return tensor_nbytes(value) if isinstance(value, torch.Tensor) else 0


def _first_tensor(values: Any) -> torch.Tensor | None:
    for value in _iter_tensors(values):
        return value
    return None


def _sparse_source_result_bytes(result: Any) -> int:
    return count_unique_tensor_bytes(result)


def _scatter_update_bytes(args: tuple[Any, ...], result: Any) -> int:
    if len(args) >= 4 and isinstance(args[3], torch.Tensor):
        return min(tensor_nbytes(args[3]), count_unique_tensor_bytes(result))
    index = args[2] if len(args) >= 3 else None
    if isinstance(index, torch.Tensor):
        out = _first_tensor(result)
        elem_size = out.element_size() if out is not None else 1
        return min(index.numel() * elem_size, count_unique_tensor_bytes(result))
    return count_unique_tensor_bytes(result)


def _index_put_update_bytes(args: tuple[Any, ...], result: Any) -> int:
    values = args[2] if len(args) >= 3 else None
    if isinstance(values, torch.Tensor):
        return min(tensor_nbytes(values), count_unique_tensor_bytes(result))
    return count_unique_tensor_bytes(result)


def _is_view_op(name: str) -> bool:
    return name in {
        "aten.alias.default",
        "aten.as_strided.default",
        "aten.detach.default",
        "aten.expand.default",
        "aten.flatten.using_ints",
        "aten.permute.default",
        "aten.reshape.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.default",
        "aten.squeeze.dim",
        "aten.t.default",
        "aten.transpose.int",
        "aten.unsqueeze.default",
        "aten.view.default",
        "aten._unsafe_view.default",
    }


def _is_sparse_source_op(name: str) -> bool:
    return name in {
        "aten.embedding.default",
        "aten.gather.default",
        "aten.index.Tensor",
        "aten.index_select.default",
        "aten.take.default",
    }


class _EffectiveByteCounter(TorchDispatchMode):
    """Track effective memory bandwidth for a fused region.

    Key design decisions:
    - We keep references to all intermediate tensors (_keepalive) to prevent
      Python from recycling id() values, which would cause spurious matches
      in our id-keyed dictionaries.
    - View/slice ops that narrow a tensor only charge the narrowed portion,
      not the full source tensor. We track per-root the maximum bytes actually
      accessed through any view chain (capped at full tensor size).
    - Sparse source ops (embedding, gather, index, index_select) charge the
      result size as the read from the source, not the full source tensor.
    - Sparse update ops (index_put, scatter) charge the update size as the
      write, not the full output tensor.
    """

    def __init__(self):
        super().__init__()
        self._keepalive: list[Any] = []
        self._produced: set[int] = set()
        self._views_of: dict[int, int] = {}
        # For each root input tensor id -> total tensor bytes
        self._input_bytes: dict[int, int] = {}
        # For each root input -> actual bytes accessed (max of view sizes)
        self._input_accessed_bytes: dict[int, int] = {}
        self._full_read_roots: set[int] = set()
        self._sparse_read_bytes: dict[int, int] = {}
        self._output_write_bytes: dict[int, int] = {}

    def _root(self, tid: int) -> int:
        seen = set()
        while tid in self._views_of and tid not in seen:
            seen.add(tid)
            tid = self._views_of[tid]
        return tid

    def _track_external_input(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        tid = self._root(id(value))
        if tid not in self._produced and tid not in self._input_bytes:
            self._input_bytes[tid] = tensor_nbytes(value)
            # Start with 0 accessed bytes; actual access is recorded by
            # _mark_full_read or _mark_sparse_read
            self._input_accessed_bytes[tid] = 0

    def _mark_full_read(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        root = self._root(id(value))
        if root in self._input_bytes:
            self._full_read_roots.add(root)
            # Update accessed bytes to the actual size being read
            accessed = tensor_nbytes(value)
            self._input_accessed_bytes[root] = max(
                self._input_accessed_bytes.get(root, 0), accessed
            )

    def _mark_sparse_read(self, value: Any, nbytes: int) -> None:
        if not isinstance(value, torch.Tensor):
            return
        root = self._root(id(value))
        if root in self._input_bytes:
            self._sparse_read_bytes[root] = self._sparse_read_bytes.get(root, 0) + nbytes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _op_name(func)
        operands = (args, kwargs)

        for tensor in _iter_tensors(operands):
            self._track_external_input(tensor)

        result = func(*args, **kwargs)

        # Keep all intermediate tensors alive to prevent id() reuse
        for tensor in _iter_tensors(result):
            self._keepalive.append(tensor)

        for tensor in _iter_tensors(result):
            self._produced.add(id(tensor))

        if _is_view_op(name) and args and isinstance(args[0], torch.Tensor):
            for tensor in _iter_tensors(result):
                self._views_of[id(tensor)] = self._root(id(args[0]))
            return result

        sparse_source_ids = set()
        if _is_sparse_source_op(name) and args and isinstance(args[0], torch.Tensor):
            sparse_source_ids.add(id(args[0]))
            self._mark_sparse_read(args[0], _sparse_source_result_bytes(result))

        if name.startswith("aten.scatter.") or name.startswith("aten.scatter_add.") or name.startswith("aten.scatter_reduce."):
            update_bytes = _scatter_update_bytes(args, result)
            if len(args) >= 4 and isinstance(args[3], torch.Tensor):
                sparse_source_ids.add(id(args[3]))
                self._mark_sparse_read(args[3], update_bytes)
            for tensor in _iter_tensors(result):
                self._output_write_bytes[id(tensor)] = update_bytes

        if name == "aten.index_put.default":
            update_bytes = _index_put_update_bytes(args, result)
            if len(args) >= 3 and isinstance(args[2], torch.Tensor):
                sparse_source_ids.add(id(args[2]))
                self._mark_sparse_read(args[2], update_bytes)
            for tensor in _iter_tensors(result):
                self._output_write_bytes[id(tensor)] = update_bytes

        for tensor in _iter_tensors(operands):
            if id(tensor) in sparse_source_ids:
                continue
            self._mark_full_read(tensor)

        return result

    def _read_bytes(self) -> int:
        total = 0
        for root, full_bytes in self._input_bytes.items():
            if root in self._full_read_roots:
                # Charge the actual accessed portion, not the full tensor
                accessed = min(self._input_accessed_bytes.get(root, full_bytes), full_bytes)
                total += accessed
            else:
                total += min(self._sparse_read_bytes.get(root, 0), full_bytes)
        return total

    def _output_bytes(self, outputs: Any) -> int:
        seen = set()
        total = 0
        for tensor in _iter_tensors(outputs):
            key = tensor_alias_key(tensor)
            if key in seen:
                continue
            seen.add(key)
            root = self._root(id(tensor))
            if root in self._input_bytes and id(tensor) not in self._produced:
                continue
            if id(tensor) in self._output_write_bytes:
                # Only charge sparse write bytes if this tensor IS the
                # sparse-update result (same shape). If this is a downstream
                # reduction of that result, charge actual output size.
                sparse_bytes = self._output_write_bytes[id(tensor)]
                actual_bytes = tensor_nbytes(tensor)
                # If actual output is much smaller than the sparse write
                # (e.g., a reduction of an index_put result), charge actual
                total += min(sparse_bytes, actual_bytes)
            else:
                total += tensor_nbytes(tensor)
        return total

    def total(self, outputs: Any) -> int:
        return self._read_bytes() + self._output_bytes(outputs)
