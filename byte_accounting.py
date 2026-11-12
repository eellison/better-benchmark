"""Logical byte accounting for SOL benchmark baselines.

The accounting in this module is deliberately a *useful/logical* byte model:
original input storage read by a fused region plus storage written for its
observable outputs.  It is not a prediction of every memory transaction.  The
public result separately reports the semantic exceptions that matter to a
dense-memory SOL comparison: indirect reads/writes and atomic writes.  Strides,
kernel count, reductions, compute mix and cache behavior belong to the
per-hardware oracle, not to this count.

Only external inputs are charged as reads, so an indirect read through a dense
intermediate (``gather(repeat(x))``, ``index(clone(x))``) charges all of ``x``
even where Inductor would fuse the intermediate away.  Atomic accumulation
(``index_put_(accumulate=True)``, ``scatter_add_``) charges a full read of the
destination: conservative, since only the updated rows are touched.

Bytes are tracked as unions of byte runs per storage, so overlapping views
are counted once, disjoint views add up, and stride-0 (expanded) inputs are
charged for their distinct elements.  A view with more than
``_MAX_EXACT_INTERVALS`` runs is charged its logical size, capped by the
union of such views' spans, and sets ``access_overlap_estimated``.  Bytes the
region already overwrote are not charged again when it reads them back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

import torch
from torch.utils import _pytree
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


INDIRECT_READ = "indirect_read"
INDIRECT_WRITE = "indirect_write"
ATOMIC_WRITE = "atomic_write"
ACCESS_OVERLAP_ESTIMATED = "access_overlap_estimated"

# Reasons a plain dense-memory roofline may not apply to a region.
MITIGATION_FACTOR_CODES = frozenset({INDIRECT_READ, INDIRECT_WRITE, ATOMIC_WRITE})


@dataclass(frozen=True)
class ByteAccounting:
    """Logical byte count plus the factors that qualify it."""

    logical_read_bytes: int
    logical_write_bytes: int
    total_bytes: int
    factors: tuple[str, ...]

    @property
    def accounting_estimated(self) -> bool:
        """Whether a pathological storage layout required a conservative union."""
        return ACCESS_OVERLAP_ESTIMATED in self.factors

    @property
    def mitigation_factors(self) -> tuple[str, ...]:
        """Semantic reasons a dense-memory roofline may not apply directly."""
        return tuple(
            factor for factor in self.factors if factor in MITIGATION_FACTOR_CODES
        )

    def to_result_fields(self) -> dict[str, Any]:
        """Return JSON-friendly fields for benchmark result rows."""
        return {
            "sol_accounting_version": "logical-v2",
            "logical_read_bytes": self.logical_read_bytes,
            "logical_write_bytes": self.logical_write_bytes,
            "total_bytes": self.total_bytes,
            "sol_accounting_estimated": self.accounting_estimated,
            "sol_mitigation_factors": list(self.mitigation_factors),
        }


def sol_result_fields(
    accounting: ByteAccounting,
    *,
    proxy_bytes: int,
    proxy_op: str,
    sol_us: float | None = None,
) -> dict[str, Any]:
    """Return the standard per-shape SOL accounting/result fields."""
    fields = accounting.to_result_fields()
    fields.update(
        {
            "sol_proxy_bytes": proxy_bytes,
            "sol_proxy_read_bytes": proxy_bytes // 2,
            "sol_proxy_write_bytes": proxy_bytes - proxy_bytes // 2,
            "sol_proxy_op": proxy_op,
            "sol_proxy_floor_applied": accounting.total_bytes < 2_048,
            "logical_to_proxy_bytes_ratio": (
                accounting.total_bytes / proxy_bytes if proxy_bytes else None
            ),
            "sol_proxy_bandwidth_gbs": (
                proxy_bytes / (sol_us * 1_000) if sol_us else None
            ),
        }
    )
    return fields


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def count_unique_tensor_bytes(values: Any) -> int:
    """Count tensor bytes in a nested value, unioning overlapping aliases."""
    by_storage: dict[tuple[Any, ...], _StorageAccess] = {}
    for tensor in _iter_tensors(values):
        key, capacity = _storage_key_and_capacity(tensor)
        access = by_storage.setdefault(key, _StorageAccess(capacity))
        access.add_tensor(tensor)
    return sum(access.nbytes() for access in by_storage.values())


def count_bytes_naive(inputs: Any, outputs: Any) -> int:
    """Count all tensor input bytes plus unique output storage bytes."""
    return sum(tensor_nbytes(t) for t in _iter_tensors(inputs)) + count_unique_tensor_bytes(outputs)


def analyze_bytes_effective(mod, inputs: Iterable[Any]) -> ByteAccounting:
    """Return a structured logical-byte audit for one execution of ``mod``."""
    inputs = tuple(inputs)
    with _EffectiveByteCounter(inputs) as counter:
        with torch.no_grad():
            outputs = mod(*inputs)
    return counter.breakdown(outputs)


def count_bytes_effective(mod, inputs: Iterable[Any]) -> int:
    """Return logical input reads plus final output writes for a repro module."""
    return analyze_bytes_effective(mod, inputs).total_bytes


def _iter_tensors(values: Any):
    for value in _pytree.tree_leaves(values):
        if isinstance(value, torch.Tensor):
            yield value


def _op_name(func) -> str:
    return str(func)


def _storage_key_and_capacity(tensor: torch.Tensor) -> tuple[tuple[Any, ...], int]:
    """Identify a storage without conflating distinct empty/meta storages."""
    try:
        storage = tensor.untyped_storage()
        identity = getattr(storage, "_cdata", None)
        if identity is None:
            identity = storage.data_ptr()
        capacity = storage.nbytes()
    except RuntimeError:
        identity = ("tensor", id(tensor))
        capacity = tensor_nbytes(tensor)
    device = tensor.device
    return (device.type, device.index, identity), capacity


# Above this many storage runs a view's byte count is estimated instead of
# enumerated exactly (each run costs two int64s while counting).
_MAX_EXACT_INTERVALS = 1 << 22


def _tensor_descriptor(tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.layout == torch.strided:
        return (
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.element_size(),
        )
    return (id(tensor), tensor_nbytes(tensor))


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Smallest [start, end) byte range of the storage that holds the tensor."""
    if tensor.layout != torch.strided or tensor.numel() == 0:
        return 0, tensor_nbytes(tensor)
    minimum = maximum = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (size - 1) * stride
        minimum += min(0, delta)
        maximum += max(0, delta)
    element_size = tensor.element_size()
    return minimum * element_size, (maximum + 1) * element_size


def _tensor_runs(tensor: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """Return the byte starts of the tensor's contiguous storage runs and the
    length of each run.

    ``None`` means exact enumeration would exceed ``_MAX_EXACT_INTERVALS``;
    the caller then estimates the byte count and sets the estimated flag.
    """
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.int64), 0
    if tensor.layout != torch.strided:
        return None

    sizes = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    element_size = tensor.element_size()

    # Collapse a row-major contiguous suffix into one run per outer
    # coordinate.  Size-one and stride-zero dimensions do not multiply the
    # number of distinct runs.
    suffix_start = len(sizes)
    block_elements = 1
    for dim in range(len(sizes) - 1, -1, -1):
        size = sizes[dim]
        if size <= 1:
            suffix_start = dim
            continue
        if strides[dim] == block_elements:
            block_elements *= size
            suffix_start = dim
            continue
        break

    outer_dims = [
        dim
        for dim in range(suffix_start)
        if sizes[dim] > 1 and strides[dim] != 0
    ]
    if prod(sizes[dim] for dim in outer_dims) > _MAX_EXACT_INTERVALS:
        # A permutation of a dense view can have millions of logical rows but
        # is still exactly one contiguous storage span.
        start, end = _byte_span(tensor)
        if end - start == tensor_nbytes(tensor):
            with _disable_current_modes():
                return torch.tensor([start], dtype=torch.int64), end - start
        return None

    with _disable_current_modes():
        offsets = torch.tensor([tensor.storage_offset()], dtype=torch.int64)
        for dim in outer_dims:
            steps = torch.arange(sizes[dim], dtype=torch.int64) * strides[dim]
            offsets = (offsets[:, None] + steps).reshape(-1)
        return offsets * element_size, block_elements * element_size


def _union_nbytes(runs: Iterable[tuple[torch.Tensor, int]]) -> int:
    """Bytes covered by the union of all runs."""
    runs = [(begin, length) for begin, length in runs if length and begin.numel()]
    if not runs:
        return 0
    with _disable_current_modes():
        starts, order = torch.sort(torch.cat([begin for begin, _ in runs]))
        ends = torch.cat([begin + length for begin, length in runs])[order]
        reach = torch.cummax(ends, 0).values
        covered_from = torch.maximum(starts, torch.cat([starts[:1], reach[:-1]]))
        return int((ends - covered_from).clamp_min(0).sum())


class _StorageAccess:
    """Union the storage runs accessed through views of one storage."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.runs: dict[tuple[Any, ...], tuple[torch.Tensor, int]] = {}
        # descriptor -> (logical bytes, byte span) of views too irregular to
        # enumerate.
        self.estimated: dict[tuple[Any, ...], tuple[int, tuple[int, int]]] = {}

    @property
    def had_estimate(self) -> bool:
        return bool(self.estimated)

    def add_tensor(self, tensor: torch.Tensor) -> None:
        descriptor = _tensor_descriptor(tensor)
        if descriptor in self.runs or descriptor in self.estimated:
            return
        runs = _tensor_runs(tensor)
        if runs is None:
            self.estimated[descriptor] = (tensor_nbytes(tensor), _byte_span(tensor))
        else:
            self.runs[descriptor] = runs

    def covers(self, tensor: torch.Tensor) -> bool:
        """Whether every byte of ``tensor`` is in an exactly known run."""
        runs = _tensor_runs(tensor)
        if runs is None or not self.runs:
            return False
        known = list(self.runs.values())
        return _union_nbytes(known + [runs]) == _union_nbytes(known)

    def nbytes(self) -> int:
        exact = _union_nbytes(self.runs.values())
        if not self.estimated:
            return exact
        # Estimated views are charged their logical bytes, but never more than
        # the union of their spans, and the total never exceeds the storage.
        with _disable_current_modes():
            spans = [
                (torch.tensor([start], dtype=torch.int64), end - start)
                for _, (start, end) in self.estimated.values()
            ]
        estimate = min(
            sum(nbytes for nbytes, _ in self.estimated.values()),
            _union_nbytes(spans),
        )
        return min(self.capacity, exact + estimate)


def _is_view_op(name: str) -> bool:
    return name in {
        "aten.alias.default",
        "aten.as_strided.default",
        "aten.detach.default",
        "aten.expand.default",
        "aten.flatten.using_ints",
        "aten.movedim.int",
        "aten.narrow.default",
        "aten.permute.default",
        "aten.reshape.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.squeeze.default",
        "aten.squeeze.dim",
        "aten.t.default",
        "aten.transpose.int",
        "aten.unflatten.int",
        "aten.unsqueeze.default",
        "aten.view.default",
        "aten._unsafe_view.default",
    }


def _returns_alias(func) -> bool:
    """Recognize view-like ops, including multi-output split/unbind variants."""
    try:
        return any(result.alias_info is not None for result in func._schema.returns)
    except (AttributeError, RuntimeError):
        return False


def _results_alias_operands(result: Any, operands: Any) -> bool:
    """Check the runtime alias decision of view-or-copy operators."""
    operand_storages = {
        _storage_key_and_capacity(tensor)[0] for tensor in _iter_tensors(operands)
    }
    result_tensors = tuple(_iter_tensors(result))
    return bool(result_tensors) and all(
        _storage_key_and_capacity(tensor)[0] in operand_storages
        for tensor in result_tensors
    )


def _is_sparse_source_op(name: str) -> bool:
    return name in {
        "aten._unsafe_index.Tensor",
        "aten._unsafe_index.Tensor_hacked_twin",
        "aten.embedding.default",
        "aten.gather.default",
        "aten.index.Tensor",
        "aten.index_select.default",
        "aten.take.default",
    }


def _is_scatter_op(name: str) -> bool:
    return (
        name.startswith("aten.scatter.")
        or name.startswith("aten.scatter_")
        or name.startswith("aten.scatter_add.")
        or name.startswith("aten.scatter_add_")
        or name.startswith("aten.scatter_reduce.")
        or name.startswith("aten.scatter_reduce_")
    )


def _is_index_put_op(name: str) -> bool:
    return name in {"aten.index_put.default", "aten.index_put_.default"}


def _is_index_update_op(name: str) -> bool:
    return (
        name == "aten._unsafe_masked_index_put_accumulate.default"
        or name.startswith("aten.index_add.")
        or name.startswith("aten.index_add_")
        or name.startswith("aten.index_reduce.")
        or name.startswith("aten.index_reduce_")
    )


def _shape_only_operand_indices(name: str) -> tuple[int, ...]:
    if name in {
        "aten._adaptive_avg_pool2d_backward.default",
        "aten.avg_pool2d_backward.default",
        "aten.avg_pool3d_backward.default",
    }:
        return (1,)
    return ()


def _is_full_overwrite_mutation(name: str) -> bool:
    """Whether an in-place op replaces every destination element.

    These operators do not need the previous destination value.  Keep this
    list deliberately narrow: partial/masked updates must continue to charge
    the destination read unless their accessed region is modeled explicitly.
    """
    return name.startswith(
        (
            "aten.copy_.",
            "aten.fill_.",
            "aten.zero_.",
        )
    )


def _is_metadata_inplace_op(name: str) -> bool:
    """In-place ops that change only the tensor's metadata, never its data."""
    return name.split(".")[1] in {
        "as_strided_",
        "detach_",
        "squeeze_",
        "swapaxes_",
        "swapdims_",
        "t_",
        "transpose_",
        "unsqueeze_",
    }


def _is_atomic_op(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if (
        "scatter_add" in name
        or "scatter_reduce" in name
        or name == "aten._unsafe_masked_index_put_accumulate.default"
    ):
        return True
    if _is_scatter_op(name):
        # scatter(..., reduce=...) accumulates into the destination.
        return name.endswith((".reduce", ".value_reduce"))
    if _is_index_put_op(name):
        if "accumulate" in kwargs:
            return bool(kwargs["accumulate"])
        return len(args) >= 4 and bool(args[3])
    return name.startswith(("aten.index_add", "aten.index_reduce"))


def _mutated_arguments(func, args, kwargs) -> list[tuple[torch.Tensor, bool]]:
    """Return (tensor, is_out) for every tensor argument the op writes."""
    try:
        schema_arguments = func._schema.arguments
    except (AttributeError, RuntimeError):
        return []
    mutated = []
    for index, argument in enumerate(schema_arguments):
        if argument.alias_info is None or not argument.alias_info.is_write:
            continue
        value = args[index] if index < len(args) else kwargs.get(argument.name)
        mutated.extend((tensor, argument.is_out) for tensor in _iter_tensors(value))
    return mutated


def _index_put_write_elements(target: torch.Tensor, indices: Any) -> int | None:
    """Number of destination elements ``target.index_put_(indices, ...)`` writes.

    Values broadcast to the indexed shape, so the write size comes from the
    indices, not from ``values``.  ``None`` when a boolean mask's population
    is not available (e.g. meta tensors).
    """
    if not isinstance(indices, (list, tuple)):
        return None
    index_shapes = []
    remaining = 1
    dim = 0
    for index in indices:
        if index is None:
            remaining *= target.shape[dim]
            dim += 1
        elif index.dtype in (torch.bool, torch.uint8):
            try:
                index_shapes.append((int(index.count_nonzero()),))
            except (RuntimeError, NotImplementedError):
                return None
            dim += index.dim()
        else:
            index_shapes.append(tuple(index.shape))
            dim += 1
    remaining *= prod(target.shape[dim:])
    if not index_shapes:
        return target.numel()
    return prod(torch.broadcast_shapes(*index_shapes)) * remaining


class _EffectiveByteCounter(TorchDispatchMode):
    """Track storage-level input reads, observable writes, and SOL factors."""

    def __init__(self, inputs: Iterable[Any] = ()):
        super().__init__()
        self._keepalive: list[Any] = []
        self._produced: set[int] = set()
        self._input_storage: dict[tuple[Any, ...], int] = {}
        self._read_access: dict[tuple[Any, ...], _StorageAccess] = {}
        self._sparse_read_bytes: dict[tuple[Any, ...], int] = {}
        self._mutation_access: dict[tuple[Any, ...], _StorageAccess] = {}
        self._sparse_write_bytes: dict[tuple[Any, ...], int] = {}
        self._factors: set[str] = set()

        # Register explicit inputs up front so separately passed views sharing
        # storage can be unioned, while still charging only inputs actually used.
        for tensor in _iter_tensors(inputs):
            self._register_external_input(tensor)

    def _register_external_input(self, tensor: torch.Tensor) -> None:
        key, capacity = _storage_key_and_capacity(tensor)
        self._input_storage[key] = max(self._input_storage.get(key, 0), capacity)

    def _track_external_input(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor) or id(value) in self._produced:
            return
        self._register_external_input(value)

    def _input_key(self, tensor: torch.Tensor) -> tuple[Any, ...] | None:
        key, _ = _storage_key_and_capacity(tensor)
        return key if key in self._input_storage else None

    def _overwritten_in_region(self, key: tuple[Any, ...], tensor: torch.Tensor) -> bool:
        """Whether the region already replaced these bytes, so reading them is
        not a read of the input."""
        access = self._mutation_access.get(key)
        return access is not None and access.covers(tensor)

    def _mark_full_read(self, value: Any) -> None:
        if not isinstance(value, torch.Tensor):
            return
        key = self._input_key(value)
        if key is None or self._overwritten_in_region(key, value):
            return
        access = self._read_access.setdefault(key, _StorageAccess(self._input_storage[key]))
        access.add_tensor(value)
        if access.had_estimate:
            self._factors.add(ACCESS_OVERLAP_ESTIMATED)

    def _mark_sparse_read(self, value: Any, nbytes: int) -> None:
        if not isinstance(value, torch.Tensor):
            return
        key = self._input_key(value)
        if key is None or self._overwritten_in_region(key, value):
            return
        # Preserve the established useful-byte convention: an indirect source
        # cannot contribute more unique useful bytes than its backing storage.
        capacity = self._input_storage[key]
        self._sparse_read_bytes[key] = min(
            capacity, self._sparse_read_bytes.get(key, 0) + nbytes
        )

    def _mark_mutation(self, tensor: torch.Tensor) -> None:
        key, capacity = _storage_key_and_capacity(tensor)
        self._mutation_access.setdefault(key, _StorageAccess(capacity)).add_tensor(tensor)

    def _mark_sparse_mutation(self, tensor: torch.Tensor, nbytes: int | None) -> None:
        """Charge an indexed write; repeated updates of one storage add up."""
        if nbytes is None:
            self._factors.add(ACCESS_OVERLAP_ESTIMATED)
            nbytes = tensor_nbytes(tensor)
        key, capacity = _storage_key_and_capacity(tensor)
        self._sparse_write_bytes[key] = min(
            capacity, self._sparse_write_bytes.get(key, 0) + nbytes
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _op_name(func)
        operands = (args, kwargs)

        for tensor in _iter_tensors(operands):
            self._track_external_input(tensor)

        result = func(*args, **kwargs)
        result_tensors = tuple(_iter_tensors(result))
        self._keepalive.extend(result_tensors)
        self._produced.update(id(tensor) for tensor in result_tensors)

        mutated = _mutated_arguments(func, args, kwargs)
        if mutated and _is_metadata_inplace_op(name):
            return result
        may_return_alias = _is_view_op(name) or _returns_alias(func)
        if (
            may_return_alias
            and not mutated
            and _results_alias_operands(result, operands)
        ):
            return result

        sparse_source_ids: set[int] = set()
        if _is_sparse_source_op(name) and args and isinstance(args[0], torch.Tensor):
            sparse_source_ids.add(id(args[0]))
            self._mark_sparse_read(args[0], count_unique_tensor_bytes(result))
            self._factors.add(INDIRECT_READ)

        scatter = _is_scatter_op(name)
        index_put = _is_index_put_op(name)
        index_update = _is_index_update_op(name)
        atomic = _is_atomic_op(name, args, kwargs)

        if scatter or index_put or index_update:
            self._factors.add(INDIRECT_WRITE)
        if atomic:
            self._factors.add(ATOMIC_WRITE)

        # Destinations whose previous value the op does not need: out=
        # arguments, full overwrites (copy_/fill_/zero_) and non-accumulating
        # scatter/index_put.
        overwrite_only = _is_full_overwrite_mutation(name) or (
            (scatter or index_put) and not atomic
        )
        skip_read_ids = set(sparse_source_ids)
        skip_read_ids.update(
            id(args[index])
            for index in _shape_only_operand_indices(name)
            if index < len(args) and isinstance(args[index], torch.Tensor)
        )
        skip_read_ids.update(
            id(tensor) for tensor, is_out in mutated if is_out or overwrite_only
        )

        # Reads first: a read-modify-write reads the destination before the
        # region's own write replaces it.
        for tensor in _iter_tensors(operands):
            if id(tensor) not in skip_read_ids:
                self._mark_full_read(tensor)

        # In-place scatter/index_put write only the indexed destinations;
        # every other mutation writes its whole argument.
        for target, is_out in mutated:
            if is_out or not (scatter or index_put):
                self._mark_mutation(target)
            elif scatter:
                index = args[2] if len(args) >= 3 else kwargs.get("index")
                elements = index.numel() if isinstance(index, torch.Tensor) else None
                self._mark_sparse_mutation(
                    target, None if elements is None else elements * target.element_size()
                )
            else:
                elements = _index_put_write_elements(target, args[1] if len(args) >= 2 else None)
                self._mark_sparse_mutation(
                    target, None if elements is None else elements * target.element_size()
                )

        return result

    def _read_bytes(self) -> int:
        keys = set(self._read_access) | set(self._sparse_read_bytes)
        total = 0
        for key in keys:
            dense_bytes = self._read_access[key].nbytes() if key in self._read_access else 0
            sparse_bytes = self._sparse_read_bytes.get(key, 0)
            total += min(self._input_storage[key], dense_bytes + sparse_bytes)
        return total

    def _write_bytes(self, outputs: Any) -> int:
        output_tensors = tuple(_iter_tensors(outputs))
        output_keys = {_storage_key_and_capacity(t)[0] for t in output_tensors}
        writes: dict[tuple[Any, ...], _StorageAccess] = dict(self._mutation_access)

        for tensor in output_tensors:
            key, capacity = _storage_key_and_capacity(tensor)
            if key in self._input_storage:
                # A returned view of an input is covered by the input's
                # mutation if it has one, and is metadata otherwise.
                continue
            writes.setdefault(key, _StorageAccess(capacity)).add_tensor(tensor)

        total = 0
        for key in set(writes) | set(self._sparse_write_bytes):
            # A mutated buffer that is neither an input nor returned lives only
            # inside the region; a fused kernel never writes it to memory.
            if key not in self._input_storage and key not in output_keys:
                continue
            access = writes.get(key)
            dense = access.nbytes() if access is not None else 0
            capacity = access.capacity if access is not None else self._input_storage[key]
            if access is not None and access.had_estimate:
                self._factors.add(ACCESS_OVERLAP_ESTIMATED)
            total += min(capacity, dense + self._sparse_write_bytes.get(key, 0))
        return total

    def breakdown(self, outputs: Any) -> ByteAccounting:
        logical_read_bytes = self._read_bytes()
        logical_write_bytes = self._write_bytes(outputs)
        return ByteAccounting(
            logical_read_bytes=logical_read_bytes,
            logical_write_bytes=logical_write_bytes,
            total_bytes=logical_read_bytes + logical_write_bytes,
            factors=tuple(sorted(self._factors)),
        )
