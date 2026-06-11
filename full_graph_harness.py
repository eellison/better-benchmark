"""Helpers for executing saved full_graph_*.py artifacts.

The saved full graphs are print_readable-style Python modules.  They contain a
torch.nn.Module subclass with tensor metadata in forward annotations, but no
canonical make_inputs() helper.  This module reconstructs runnable inputs from
those annotations so the benchmark runner can time the whole graph directly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FullGraphDefinition:
    path: Path
    graph_cls: type
    input_specs: list[dict[str, Any]]
    tensor_attrs: dict[str, dict[str, Any]]
    forward_takes_no_inputs: bool
    metadata: dict[str, Any]


_DTYPE_SHORT_NAMES = {
    "f32": "float32",
    "f16": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "b8": "bool",
    "f64": "float64",
    "c64": "complex64",
    "c128": "complex128",
    "f8e4m3fn": "float8_e4m3fn",
    "f8e5m2": "float8_e5m2",
}


def _torch_dtype(dtype_name: str):
    import torch

    return getattr(torch, dtype_name, torch.float32)


def _dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def _default_generator_for_dtype(
    dtype_name: str,
    shape: tuple[int, ...] | list[int] = (),
) -> dict[str, Any]:
    if dtype_name == "bool":
        return {"kind": "randint", "low": 0, "high": 2}
    if dtype_name.startswith("int") or dtype_name.startswith("uint"):
        high = max([int(dim) for dim in shape], default=100)
        return {"kind": "index", "low": 0, "high": max(high, 1)}
    return {"kind": "randn"}


def _is_integer_or_bool_dtype_name(dtype_name: str) -> bool:
    return (
        dtype_name == "bool"
        or dtype_name.startswith("int")
        or dtype_name.startswith("uint")
    )


def _default_generator_for_attr_dtype(dtype_name: str) -> dict[str, Any]:
    # Annotation-only tensor constants have no payload. Zero is safer than
    # random bounded indices for constants that feed indexing/scatter paths.
    if _is_integer_or_bool_dtype_name(dtype_name):
        return {"kind": "constant", "value": 0}
    return {"kind": "randn"}


def _concrete_int(value: Any, *, default: int = 32) -> int:
    try:
        if hasattr(value, "node") and hasattr(value.node, "hint"):
            return int(value.node.hint)
        return int(value)
    except Exception:
        return default


def placeholder_info_from_gm(gm: Any) -> dict[str, dict]:
    """Build the placeholder_info shape/dtype map used for input generators."""
    import torch

    placeholder_info = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = (node.meta or {}).get("val")
        if torch.is_tensor(val):
            placeholder_info[node.name] = {
                "shape": [_concrete_int(s) for s in val.shape],
                "stride": [_concrete_int(s) for s in val.stride()]
                if not val.is_contiguous()
                else [],
                "dtype": str(val.dtype),
                "device": str(val.device),
            }
        elif isinstance(val, (torch.SymInt, torch.SymFloat)):
            placeholder_info[node.name] = {
                "shape": [],
                "stride": [],
                "dtype": "symint",
                "device": "cpu",
                "hint": _concrete_int(val),
            }
    return placeholder_info


def infer_index_bounds_from_gm(
    gm: Any,
    placeholder_info: dict[str, dict],
    constants_out: dict[str, int] | None = None,
) -> dict[str, int]:
    """Infer valid index bounds for integer placeholders by inspecting consumers.

    `constants_out` (optional dict, filled in place) receives placeholders
    whose only safe generation is a CONSTANT value rather than a range —
    e.g. maxpool offset tensors, where the window-center offset is the one
    value in-bounds for every window under padding (random offsets put edge
    windows out of range -> device-side assert in the consuming scatter).
    """
    import torch

    index_consumers = {
        torch.ops.aten.scatter.src: (0, 1),
        torch.ops.aten.scatter.value: (0, 1),
        torch.ops.aten.scatter_add.default: (0, 1),
        torch.ops.aten.gather.default: (0, 1),
        torch.ops.aten.index_put.default: (0, None),
        torch.ops.aten.index.Tensor: (0, None),
        torch.ops.aten.index_select.default: (0, 1),
        torch.ops.aten.embedding.default: (0, None),
    }
    passthrough_ops = {
        torch.ops.aten.clone.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.select.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.where.self,
    }

    bounds = {}
    constants = constants_out if constants_out is not None else {}
    ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    def _node_shape(n):
        if not hasattr(n, "meta"):
            return None
        val = n.meta.get("val")
        if isinstance(val, torch.Tensor):
            return list(val.shape)
        if n.op == "call_function" and n.target in {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.full.default,
        }:
            if n.args and isinstance(n.args[0], (list, tuple)):
                return list(n.args[0])
        return None

    for name, node in ph_nodes.items():
        info = placeholder_info.get(name, {})
        dtype = info.get("dtype", "")
        if "int" not in dtype:
            continue

        if "int8" in dtype:
            # Maxpool OFFSET tensors (offset-within-window). Random offsets
            # are NOT uniformly valid: offsets_to_indices converts them to
            # flat indices relative to each window's position, and under
            # padding the edge windows go negative / past the dim for most
            # values (offset 0 at a padded edge -> index -113) — the
            # consuming scatter_add then device-side-asserts and poisons
            # the CUDA context (the wave-1 torchbench train assert class).
            # The WINDOW-CENTER offset is in-bounds for every window
            # (measured: 3x3 pad1 on 112x112 -> [0, 12430] vs dim 12544),
            # so emit it via constants_out as a constant-value generator.
            for user in node.users:
                if user.op != "call_function":
                    continue
                target_name = str(user.target)
                if "max_pool_offsets_to_indices" not in target_name:
                    continue
                if len(user.args) >= 2 and isinstance(user.args[1], (list, tuple)):
                    kernel_size = user.args[1]
                    if len(kernel_size) >= 2:
                        kh, kw = int(kernel_size[0]), int(kernel_size[1])
                        center = (kh // 2) * kw + (kw // 2)
                    else:
                        center = int(kernel_size[0]) // 2
                    constants[name] = center
                    break
            if name not in constants:
                constants[name] = 4  # assume 3x3: center offset 4
            continue

        def _find_bound(start_node, max_hops=8):
            # Collect EVERY bound the value reaches and return the MIN: a
            # value consumed by several index ops must satisfy all of them
            # (Electra: ids flow through gather into the 2-row token-type
            # embedding — first-found returned the 512 gather bound and the
            # 2-row table asserted OOB, poisoning the CUDA context).
            #
            # The walk follows VALUE flow through int arithmetic too, and
            # INVERTS each transform when a consumption bound is found
            # (Longformer: position ids = (ids * mask + 1) indexes a
            # 4098-row table — the leaf bound is isqrt(4097), not 4098):
            #   add/sub const c   ->  bound - |c|
            #   mul const c>0    ->  bound // c
            #   mul TENSOR       ->  isqrt(bound) (both operands generated
            #                        by us; each < sqrt(B) keeps product < B)
            #   add TENSOR       ->  bound // 2  (same reasoning)
            #   convert/clone/.. ->  unchanged
            import math as _math

            found: list[int] = []

            def _invert(bound: int, chain) -> int:
                b = bound
                for kind, arg in reversed(chain):
                    if kind == "addc":
                        b = b - abs(int(arg))
                    elif kind == "mulc":
                        c = abs(int(arg))
                        b = b // c if c > 1 else b
                    elif kind == "mult":
                        b = int(_math.isqrt(max(b - 1, 1)))
                    elif kind == "addt":
                        b = b // 2
                    b = max(b, 1)
                return b

            _INT_ARITH = {
                torch.ops.aten.add.Tensor: "add",
                torch.ops.aten.sub.Tensor: "add",   # same inversion class
                torch.ops.aten.mul.Tensor: "mul",
            }
            _VALUE_PRESERVING = {
                torch.ops.prims.convert_element_type.default,
            }

            frontier = [(start_node, ())]
            for _hop in range(max_hops):
                next_frontier = []
                for n, chain in frontier:
                    for user in n.users:
                        if user.op != "call_function":
                            continue
                        target = user.target

                        if target == torch.ops.aten.embedding.default:
                            # only when n is the INDICES arg (args[1])
                            if len(user.args) > 1 and user.args[1] is n:
                                weight_arg = user.args[0] if user.args else None
                                if weight_arg and hasattr(weight_arg, "meta"):
                                    val = weight_arg.meta.get("val")
                                    if isinstance(val, torch.Tensor) and len(val.shape) > 0:
                                        found.append(_invert(int(val.shape[0]), chain))
                                        continue

                        if target == torch.ops.aten.index.Tensor and len(user.args) >= 2:
                            target_shape = _node_shape(user.args[0])
                            indices = user.args[1]
                            if target_shape and isinstance(indices, (list, tuple)):
                                for dim, index_node in enumerate(indices):
                                    if index_node is n and dim < len(target_shape):
                                        found.append(_invert(int(target_shape[dim]), chain))

                        if target == torch.ops.aten.gather.default:
                            if user.args and user.args[0] is n:
                                # n is gather's DATA source: values flow on.
                                next_frontier.append((user, chain))
                                continue

                        if target in index_consumers:
                            target_arg_idx, dim_arg_idx = index_consumers[target]
                            if len(user.args) > target_arg_idx:
                                target_node = user.args[target_arg_idx]
                                target_shape = _node_shape(target_node)
                                if target_shape and target_node is not n:
                                    if dim_arg_idx is not None and len(user.args) > dim_arg_idx:
                                        dim = user.args[dim_arg_idx]
                                        if isinstance(dim, int) and dim < len(target_shape):
                                            found.append(_invert(int(target_shape[dim]), chain))
                                    else:
                                        found.append(_invert(int(target_shape[0]), chain))

                        if target in passthrough_ops or target in _VALUE_PRESERVING:
                            next_frontier.append((user, chain))
                        elif target in _INT_ARITH:
                            kind = _INT_ARITH[target]
                            other = None
                            for a in user.args[:2]:
                                if a is not n:
                                    other = a
                            if isinstance(other, (int, float)):
                                next_frontier.append(
                                    (user, chain + ((kind + "c", other),)))
                            else:
                                next_frontier.append(
                                    (user, chain + ((kind + "t", None),)))
                frontier = next_frontier
                if not frontier:
                    break
            if found:
                return min(found)
            return None

        bound = _find_bound(node)
        if bound is not None:
            bounds[name] = bound

    return bounds


def infer_permutation_indices_from_gm(
    gm: Any,
    placeholder_info: dict[str, dict],
) -> dict[str, int]:
    """Find integer placeholders that must be valid permutations."""
    import torch

    permutation_sizes = {}
    ph_nodes = {n.name: n for n in gm.graph.nodes if n.op == "placeholder"}

    def _shape_from_alloc(n):
        if n.op != "call_function":
            return None
        if n.target not in {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.full.default,
        }:
            return None
        if not n.args or not isinstance(n.args[0], (list, tuple)):
            return None
        return list(n.args[0])

    iota_passthrough_ops = {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
    }

    def _iota_size(n, max_hops=4):
        if n.op != "call_function":
            return None
        if n.target != torch.ops.prims.iota.default:
            if max_hops > 0 and n.target in iota_passthrough_ops and n.args:
                return _iota_size(n.args[0], max_hops - 1)
            return None
        if n.args and isinstance(n.args[0], int):
            return int(n.args[0])
        return None

    for name, node in ph_nodes.items():
        info = placeholder_info.get(name, {})
        dtype = info.get("dtype", "")
        if "int" not in dtype:
            continue
        for user in node.users:
            if user.op != "call_function" or user.target != torch.ops.aten.index_put.default:
                continue
            if len(user.args) < 3:
                continue
            indices = user.args[1]
            if not isinstance(indices, (list, tuple)) or node not in indices:
                continue
            target_shape = _shape_from_alloc(user.args[0])
            iota_size = _iota_size(user.args[2])
            if target_shape and iota_size and target_shape[0] == iota_size:
                permutation_sizes[name] = iota_size

        for user in node.users:
            if user.op != "call_function" or user.target != torch.ops.aten.scatter.src:
                continue
            if len(user.args) < 4 or user.args[2] is not node:
                continue
            target_shape = _shape_from_alloc(user.args[0])
            iota_size = _iota_size(user.args[3])
            dim = user.args[1]
            if target_shape and iota_size and isinstance(dim, int):
                if dim < 0:
                    dim += len(target_shape)
                if 0 <= dim < len(target_shape) and target_shape[dim] == iota_size:
                    permutation_sizes[name] = iota_size
    return permutation_sizes


def _tensor_spec_from_value(
    name: str,
    value: Any,
    *,
    gen_override: dict[str, Any] | None = None,
    include_exact: bool = False,
) -> dict[str, Any]:
    dtype_name = _dtype_name(value.dtype)
    shape = [_concrete_int(dim) for dim in value.shape]
    generator = gen_override or _default_generator_for_dtype(dtype_name, shape)
    spec = {
        "kind": "tensor",
        "name": name,
        "shape": shape,
        "dtype": dtype_name,
        "stride": [_concrete_int(dim) for dim in value.stride()],
        "device": "cuda" if str(value.device).startswith("cuda") else str(value.device),
        "storage_offset": _concrete_int(value.storage_offset(), default=0),
        "generator": generator,
    }
    if generator.get("kind") in {"index", "permutation"}:
        spec["gen"] = generator
        spec["constraint_source"] = "graph_inference" if gen_override else "dtype_shape_default"
    exact = _exact_tensor_payload(value) if include_exact else None
    if exact is not None:
        spec.update(exact)
    else:
        spec["exact"] = False
        if include_exact and _is_integer_or_bool_dtype_name(dtype_name):
            spec["requires_exact"] = True
    return spec


def _is_fake_or_meta_tensor(value: Any) -> bool:
    if getattr(value, "is_meta", False):
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        if isinstance(value, FakeTensor):
            return True
    except Exception:
        pass
    return "fake_tensor" in type(value).__module__


def _exact_tensor_payload(value: Any, *, max_numel: int = 4096) -> dict[str, Any] | None:
    if _is_fake_or_meta_tensor(value):
        return None
    dtype_name = _dtype_name(value.dtype)
    if value.numel() > max_numel:
        return None
    if not (
        dtype_name == "bool"
        or dtype_name.startswith("int")
        or dtype_name.startswith("uint")
        or value.numel() <= 16
    ):
        return None
    try:
        detached = value.detach().cpu().contiguous()
    except Exception:
        return None
    return {
        "exact": True,
        "data": detached.reshape(-1).tolist(),
    }


def _scalar_spec_from_value(name: str, value: Any) -> dict[str, Any]:
    import torch

    if isinstance(value, (torch.SymInt, torch.SymFloat)):
        return {"kind": "symint", "name": name, "value": _concrete_int(value)}
    if isinstance(value, (int, float, bool)):
        return {"kind": "scalar", "name": name, "value": value}
    return {"kind": "scalar", "name": name, "value": 1}


def graph_constraints_from_gm(
    gm: Any,
    *,
    index_bounds: dict[str, int] | None = None,
    permutation_indices: dict[str, int] | None = None,
    observed_stats: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Extract replay constraints from an FX GraphModule.

    The exporter writes this next to full_graph_*.py so newly added model
    graphs do not depend solely on parsing print_readable annotations.

    Bound hierarchy (settled 2026-06-10):
      1. graph_inference: known consumer patterns (embedding, gather, scatter)
      2. observed: fallback from real execution stats (high = observed.max + 1)
      3. default_unobserved: ONLY when observation was impossible (should not
         occur for new captures)
    """
    import torch

    index_bounds = index_bounds or {}
    permutation_indices = permutation_indices or {}
    observed_stats = observed_stats or {}
    inputs = []
    outputs = []
    tensor_attrs = {}
    for node in gm.graph.nodes:
        value = (node.meta or {}).get("val")
        if node.op == "placeholder":
            if torch.is_tensor(value):
                gen_override = None
                constraint_source = None
                if node.name in permutation_indices:
                    gen_override = {
                        "kind": "permutation",
                        "size": int(permutation_indices[node.name]),
                    }
                    constraint_source = "graph_inference"
                elif node.name in index_bounds:
                    gen_override = {
                        "kind": "index",
                        "low": 0,
                        "high": int(index_bounds[node.name]),
                    }
                    constraint_source = "graph_inference"
                elif node.name in observed_stats:
                    # Observed-value fallback: use observed max + 1 as bound
                    obs = observed_stats[node.name]
                    dtype_name = _dtype_name(value.dtype)
                    if _is_integer_or_bool_dtype_name(dtype_name):
                        gen_override = {
                            "kind": "index",
                            "low": 0,
                            "high": int(obs["max"]) + 1,
                        }
                        constraint_source = "observed"
                else:
                    # No inference bound and no observation
                    dtype_name = _dtype_name(value.dtype)
                    if _is_integer_or_bool_dtype_name(dtype_name):
                        constraint_source = "default_unobserved"

                spec = _tensor_spec_from_value(
                    node.name,
                    value,
                    gen_override=gen_override,
                )
                # Override constraint_source with the hierarchy decision
                if constraint_source is not None:
                    spec["constraint_source"] = constraint_source
                # Attach observed stats alongside (regardless of which source won)
                if node.name in observed_stats:
                    spec["observed"] = observed_stats[node.name]
                    # Note potential permutation: n_unique == numel on 1-D int tensor
                    obs = observed_stats[node.name]
                    shape = spec.get("shape", [])
                    if (
                        len(shape) == 1
                        and shape[0] > 0
                        and obs.get("n_unique") == shape[0]
                        and _is_integer_or_bool_dtype_name(spec.get("dtype", ""))
                        and spec.get("dtype", "") != "bool"
                    ):
                        spec["maybe_permutation"] = True
                inputs.append(spec)
            else:
                inputs.append(_scalar_spec_from_value(node.name, value))
        elif node.op == "get_attr":
            attr_value = _fetch_attr(gm, node.target)
            if torch.is_tensor(attr_value):
                tensor_attrs[str(node.target)] = _tensor_spec_from_value(
                    str(node.target),
                    attr_value,
                    include_exact=True,
                )
        elif node.op == "output":
            for idx, leaf in enumerate(_iter_tree_leaves(node.args)):
                if isinstance(leaf, torch.fx.Node):
                    leaf_value = (leaf.meta or {}).get("val")
                    if torch.is_tensor(leaf_value):
                        outputs.append(_tensor_spec_from_value(f"output_{idx}", leaf_value))
                    elif leaf_value is not None:
                        outputs.append(_scalar_spec_from_value(f"output_{idx}", leaf_value))

    payload = {
        "schema_version": 1,
        "inputs": inputs,
        "outputs": outputs,
        "tensor_attrs": tensor_attrs,
    }
    storage_groups = _placeholder_storage_groups(gm)
    if storage_groups:
        # Inputs that are VIEWS OF ONE STORAGE (packed-qkv saved views in
        # backward graphs etc). Regenerating them as independent storages
        # changes footprint/locality and is wrong under mutation — replay
        # must allocate one buffer per group and as_strided each member.
        # Detectable ONLY at capture (live fake vals share a storage; any
        # later retrace re-fabricates inputs and the identity is gone).
        payload["storage_groups"] = storage_groups
    return payload


def _placeholder_storage_groups(gm: Any) -> list[list[str]]:
    """Group placeholder names whose meta vals share one untyped storage."""
    import torch

    groups: dict[int, list[str]] = {}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = (node.meta or {}).get("val")
        if not torch.is_tensor(val):
            continue
        try:
            key = id(val.untyped_storage())
        except Exception:
            continue
        groups.setdefault(key, []).append(node.name)
    return [names for names in groups.values() if len(names) > 1]


def _fetch_attr(module: Any, target: Any) -> Any:
    value = module
    for atom in str(target).split("."):
        value = getattr(value, atom)
    return value


def _iter_tree_leaves(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tree_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tree_leaves(item)
    else:
        yield value


def write_full_graph_metadata(
    graph_path: str | Path,
    gm: Any,
    *,
    extra: dict[str, Any] | None = None,
    index_bounds: dict[str, int] | None = None,
    permutation_indices: dict[str, int] | None = None,
    observed_stats: dict[str, dict] | None = None,
) -> Path:
    graph_path = Path(graph_path)
    payload = graph_constraints_from_gm(
        gm,
        index_bounds=index_bounds,
        permutation_indices=permutation_indices,
        observed_stats=observed_stats,
    )
    payload["graph"] = graph_path.name
    if extra:
        payload.update(extra)
    # Serialize inputs/outputs/tensor_attrs in the compact shared encoding
    # (input_codec) — schema_version 2. The verbose dict-per-tensor form
    # remains the in-memory representation; loaders inflate on read.
    from input_codec import compact_from_spec

    payload["schema_version"] = 2
    for key in ("inputs", "outputs"):
        if payload.get(key):
            payload[key] = [compact_from_spec(s, include_name=True)
                            for s in payload[key]]
    if payload.get("tensor_attrs"):
        payload["tensor_attrs"] = {
            name: compact_from_spec(s)
            for name, s in payload["tensor_attrs"].items()
        }
    meta_path = graph_path.with_suffix(".meta.json")
    meta_path.write_text(_dumps_compact_entries(payload) + "\n")
    return meta_path


class _OneLine:
    """Marker: serialize this value on a single line inside indented JSON."""

    def __init__(self, value):
        self.value = value


def dumps_with_onelines(obj) -> str:
    """json.dumps(indent=2), but any _OneLine-wrapped value is emitted on a
    single line (a compact input entry exploded across 12 indented lines
    would re-create the verbosity the compact encoding exists to remove)."""
    placeholders: list[str] = []

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, _OneLine):
                placeholders.append(
                    json.dumps(o.value, separators=(",", ":")))
                return f"@@ONELINE{len(placeholders) - 1}@@"
            return super().default(o)

    text = json.dumps(obj, indent=2, cls=_Enc)
    for i, body in enumerate(placeholders):
        text = text.replace(f'"@@ONELINE{i}@@"', body)
    return text


def _dumps_compact_entries(payload: dict) -> str:
    """Sidecar serializer: inputs/outputs/tensor_attrs entries one-lined."""
    marked = dict(payload)
    for key in ("inputs", "outputs"):
        if marked.get(key):
            marked[key] = [_OneLine(e) for e in marked[key]]
    if marked.get("tensor_attrs"):
        marked["tensor_attrs"] = {
            k: _OneLine(v) for k, v in marked["tensor_attrs"].items()
        }
    return dumps_with_onelines(marked)


def _split_signature_args(sig: str) -> list[str]:
    args = []
    start = 0
    in_quote = False
    quote = ""
    for i, ch in enumerate(sig):
        if ch in {"'", '"'}:
            if in_quote and ch == quote:
                in_quote = False
                quote = ""
            elif not in_quote:
                in_quote = True
                quote = ch
        elif ch == "," and not in_quote:
            part = sig[start:i].strip()
            if part:
                args.append(part)
            start = i + 1
    part = sig[start:].strip()
    if part:
        args.append(part)
    return args


def _parse_intish(value: str, *, default: int = 32) -> int:
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        pass
    sym_match = re.fullmatch(r"Sym\(([^)]*)\)", value)
    if sym_match:
        return _parse_intish(sym_match.group(1), default=default)
    if re.fullmatch(r"[A-Za-z_]\w*", value):
        return default
    match = re.search(r"-?\d+", value)
    if match:
        return int(match.group(0))
    return default


def _symbolic_dims(shape_str: str) -> list[dict[str, Any]]:
    symbols = []
    for idx, dim in enumerate(shape_str.split(",")):
        dim = dim.strip()
        if not dim:
            continue
        try:
            int(dim)
        except ValueError:
            symbols.append({"dim": idx, "symbol": dim, "default": _parse_intish(dim)})
    return symbols


def _parse_shape(shape_str: str) -> tuple[int, ...]:
    dims = []
    for dim in shape_str.split(","):
        dim = dim.strip()
        if dim:
            dims.append(_parse_intish(dim))
    return tuple(dims)


def _parse_stride(stride_str: str | None) -> tuple[int, ...] | None:
    if stride_str is None:
        return None
    dims = []
    for dim in stride_str.split(","):
        dim = dim.strip()
        if dim:
            dims.append(_parse_intish(dim))
    return tuple(dims)


def _parse_device(device_str: str | None) -> str | None:
    if not device_str:
        return None
    device_str = device_str.strip()
    if not device_str:
        return None
    if device_str.startswith("cuda"):
        return "cuda"
    if device_str.startswith("cpu"):
        return "cpu"
    return None


def _forward_signature(content: str) -> str | None:
    match = re.search(r"def forward\(self,?\s*(.*?)\):", content, re.DOTALL)
    return match.group(1) if match else None


def forward_takes_no_inputs(content: str) -> bool:
    sig = _forward_signature(content)
    return sig is not None and not sig.strip()


def parse_full_graph_inputs(content: str) -> list[dict[str, Any]]:
    """Parse forward annotations into input specs.

    Examples handled:
      x: "f32[2, 3][3, 1]cuda:0"
      s0: "Sym(s0)"
      unannotated scalar parameters
    """
    sig = _forward_signature(content)
    if sig is None or not sig.strip():
        return []

    specs = []
    for arg in _split_signature_args(sig):
        name_match = re.match(r"(\w+)", arg)
        name = name_match.group(1) if name_match else f"arg{len(specs)}"
        annotation_match = re.match(r'\w+\s*:\s*"([^"]*)"', arg)
        if annotation_match is None:
            specs.append({"kind": "scalar", "name": name, "value": 1})
            continue

        annotation = annotation_match.group(1)
        if annotation.startswith("Sym("):
            specs.append({
                "kind": "symint",
                "name": name,
                "value": _parse_intish(annotation),
            })
            continue

        tensor_match = re.match(
            r"(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?(.*)",
            annotation,
        )
        if tensor_match is None:
            specs.append({"kind": "scalar", "name": name, "value": 1})
            continue

        dtype_token, shape_str, stride_str, device_str = tensor_match.groups()
        dtype_name = _DTYPE_SHORT_NAMES.get(dtype_token, "float32")
        shape = _parse_shape(shape_str)
        generator = _default_generator_for_dtype(dtype_name, shape)
        spec = {
            "kind": "tensor",
            "name": name,
            "shape": shape,
            "dtype": dtype_name,
            "stride": _parse_stride(stride_str),
            "device": _parse_device(device_str),
            # print_readable annotations don't carry storage_offset: None =
            # unknown (validator skips), never a false claim of 0 — packed
            # qkv views (offset 192/384/...) are real and sidecar-recorded.
            "storage_offset": None,
            "generator": generator,
        }
        if generator.get("kind") == "index":
            spec["gen"] = generator
            spec["constraint_source"] = "annotation_default"
        symbolic_dims = _symbolic_dims(shape_str)
        if symbolic_dims:
            spec["symbolic_dims"] = symbolic_dims
        specs.append({
            **spec,
        })

    return specs


def parse_full_graph_tensor_attrs(content: str) -> dict[str, dict[str, Any]]:
    """Find saved self._frozen_param* and self._tensor_constant* references."""
    attrs: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r'\w+:\s*"(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?([^"]*)"\s*=\s*'
        r"self\.(_(?:frozen_param|tensor_constant)\w*)"
    )
    for line in content.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        dtype_token, shape_str, stride_str, device_str, attr_name = match.groups()
        dtype_name = _DTYPE_SHORT_NAMES.get(dtype_token, "float32")
        shape = _parse_shape(shape_str)
        generator = _default_generator_for_attr_dtype(dtype_name)
        attrs[attr_name] = {
            "kind": "tensor",
            "name": attr_name,
            "shape": shape,
            "dtype": dtype_name,
            "stride": _parse_stride(stride_str),
            "device": _parse_device(device_str),
            "storage_offset": 0,
            "generator": generator,
        }
        if _is_integer_or_bool_dtype_name(dtype_name):
            attrs[attr_name]["requires_exact"] = True
        if generator.get("kind") == "index":
            attrs[attr_name]["gen"] = generator
    return attrs


def full_graph_meta_path(graph_path: str | Path) -> Path:
    return Path(graph_path).with_suffix(".meta.json")


def load_full_graph_sidecar(graph_path: str | Path) -> dict[str, Any]:
    meta_path = full_graph_meta_path(graph_path)
    if not meta_path.exists():
        return {}
    sidecar = json.loads(meta_path.read_text())
    # schema v2: inputs/outputs/tensor_attrs stored in the compact shared
    # encoding (input_codec); inflate to the verbose in-memory spec dicts
    # so every downstream consumer is format-agnostic.
    if sidecar.get("schema_version", 1) >= 2:
        from input_codec import spec_from_compact

        for key in ("inputs", "outputs"):
            if sidecar.get(key):
                sidecar[key] = [spec_from_compact(e) for e in sidecar[key]]
        if sidecar.get("tensor_attrs"):
            sidecar["tensor_attrs"] = {
                name: spec_from_compact(e)
                for name, e in sidecar["tensor_attrs"].items()
            }
    return sidecar


def _normalize_dtype_name(dtype_name: Any) -> str:
    return str(dtype_name).removeprefix("torch.")


def _rank(spec: dict[str, Any]) -> int | None:
    shape = spec.get("shape")
    if shape is None:
        return None
    return len(shape)


def _normalize_device_name(device: Any) -> str | None:
    if device is None:
        return None
    text = str(device)
    if not text:
        return None
    if text.startswith("cuda"):
        return "cuda"
    if text.startswith("cpu"):
        return "cpu"
    return text


def _symbolic_dim_indices(spec: dict[str, Any]) -> set[int]:
    return {
        int(item["dim"])
        for item in spec.get("symbolic_dims", [])
        if isinstance(item, dict) and "dim" in item
    }


def _validate_tensor_sidecar_spec(
    sidecar_spec: dict[str, Any],
    annotation_spec: dict[str, Any],
    label: str,
) -> None:
    sidecar_dtype = _normalize_dtype_name(sidecar_spec.get("dtype", "float32"))
    annotation_dtype = _normalize_dtype_name(annotation_spec.get("dtype", "float32"))
    if sidecar_dtype != annotation_dtype:
        raise ValueError(
            "full graph sidecar tensor dtype does not match forward annotations "
            f"for {label}: {sidecar_dtype!r} != {annotation_dtype!r}"
        )

    sidecar_shape = sidecar_spec.get("shape")
    annotation_shape = annotation_spec.get("shape")
    if sidecar_shape is not None and annotation_shape is not None:
        if len(sidecar_shape) != len(annotation_shape):
            raise ValueError(
                "full graph sidecar tensor rank does not match forward annotations "
                f"for {label}: {len(sidecar_shape)} != {len(annotation_shape)}"
            )
        symbolic_dims = _symbolic_dim_indices(annotation_spec)
        for idx, (sidecar_dim, annotation_dim) in enumerate(zip(sidecar_shape, annotation_shape)):
            if idx in symbolic_dims:
                continue
            if int(sidecar_dim) != int(annotation_dim):
                raise ValueError(
                    "full graph sidecar tensor shape does not match forward annotations "
                    f"for {label} dim {idx}: {sidecar_dim} != {annotation_dim}"
                )

    sidecar_stride = sidecar_spec.get("stride")
    annotation_stride = annotation_spec.get("stride")
    if sidecar_stride is not None and annotation_stride is not None:
        if len(sidecar_stride) != len(annotation_stride):
            raise ValueError(
                "full graph sidecar tensor stride rank does not match forward annotations "
                f"for {label}: {len(sidecar_stride)} != {len(annotation_stride)}"
            )
        for idx, (sidecar_dim, annotation_dim) in enumerate(zip(sidecar_stride, annotation_stride)):
            if int(sidecar_dim) != int(annotation_dim):
                raise ValueError(
                    "full graph sidecar tensor stride does not match forward annotations "
                    f"for {label} dim {idx}: {sidecar_dim} != {annotation_dim}"
                )

    sidecar_device = _normalize_device_name(sidecar_spec.get("device"))
    annotation_device = _normalize_device_name(annotation_spec.get("device"))
    if sidecar_device and annotation_device and sidecar_device != annotation_device:
        raise ValueError(
            "full graph sidecar tensor device does not match forward annotations "
            f"for {label}: {sidecar_device!r} != {annotation_device!r}"
        )

    sidecar_offset = sidecar_spec.get("storage_offset")
    annotation_offset = annotation_spec.get("storage_offset")
    if (
        sidecar_offset is not None
        and annotation_offset is not None
        and int(sidecar_offset) != int(annotation_offset)
    ):
        raise ValueError(
            "full graph sidecar tensor storage_offset does not match forward annotations "
            f"for {label}: {sidecar_offset} != {annotation_offset}"
        )


def _validate_sidecar_inputs(
    sidecar_inputs: list[dict[str, Any]] | None,
    annotation_inputs: list[dict[str, Any]],
) -> None:
    if sidecar_inputs is None:
        return
    if annotation_inputs and len(sidecar_inputs) != len(annotation_inputs):
        raise ValueError(
            "full graph sidecar input count does not match forward annotations: "
            f"{len(sidecar_inputs)} != {len(annotation_inputs)}"
        )
    for idx, sidecar_spec in enumerate(sidecar_inputs):
        if idx >= len(annotation_inputs):
            break
        annotation_spec = annotation_inputs[idx]
        sidecar_kind = sidecar_spec.get("kind")
        annotation_kind = annotation_spec.get("kind")
        if sidecar_kind != annotation_kind:
            raise ValueError(
                "full graph sidecar input kind does not match forward annotations "
                f"for input {idx}: {sidecar_kind!r} != {annotation_kind!r}"
            )
        sidecar_name = sidecar_spec.get("name")
        annotation_name = annotation_spec.get("name")
        if sidecar_name and annotation_name and sidecar_name != annotation_name:
            raise ValueError(
                "full graph sidecar input name does not match forward annotations "
                f"for input {idx}: {sidecar_name!r} != {annotation_name!r}"
            )
        if sidecar_kind != "tensor":
            continue
        _validate_tensor_sidecar_spec(
            sidecar_spec,
            annotation_spec,
            str(sidecar_name or idx),
        )


def _validate_sidecar_tensor_attrs(
    sidecar_attrs: dict[str, dict[str, Any]] | None,
    annotation_attrs: dict[str, dict[str, Any]],
) -> None:
    if sidecar_attrs is None:
        return
    for attr_name, sidecar_spec in sidecar_attrs.items():
        annotation_spec = annotation_attrs.get(attr_name)
        if annotation_spec is None:
            continue
        _validate_tensor_sidecar_spec(sidecar_spec, annotation_spec, attr_name)


_MODE_NAMES = {"infer", "train"}


def infer_full_graph_source(path: str | Path) -> dict[str, Any]:
    """Infer model/microbenchmark identity from repros/models layout."""
    path = Path(path)
    parts = path.parts
    try:
        models_idx = next(
            i for i in range(len(parts) - 1)
            if parts[i] == "repros" and parts[i + 1] == "models"
        )
        rel_parts = parts[models_idx + 2:]
    except StopIteration:
        rel_parts = parts

    if len(rel_parts) < 3:
        return {
            "kind": "unknown",
            "suite": None,
            "mode": None,
            "model": path.parent.name,
            "graph": path.name,
        }

    suite = rel_parts[0]
    label_parts = list(rel_parts[1:-1])
    mode = None
    if label_parts and label_parts[0] in _MODE_NAMES:
        mode = label_parts.pop(0)
    model = Path(*label_parts).as_posix() if label_parts else path.parent.name
    return {
        "kind": "microbenchmark" if suite == "genai" else "model",
        "suite": suite,
        "mode": mode,
        "model": model,
        "graph": path.name,
    }


def result_metadata(definition: FullGraphDefinition) -> dict[str, Any]:
    sidecar = definition.metadata.get("sidecar", {})
    constraints = {
        "inputs": definition.input_specs,
        "tensor_attrs": definition.tensor_attrs,
    }
    if sidecar.get("outputs") is not None:
        constraints["outputs"] = sidecar["outputs"]

    return {
        "schema_version": 1,
        "path": str(definition.path),
        "source": definition.metadata["source"],
        "constraints": constraints,
        "constraints_source": "sidecar" if sidecar else "annotations",
    }


def _strip_runnable_boilerplate(content: str) -> str:
    content = content.replace("class <lambda>", "class Repro")
    content = re.sub(
        r"^if __name__\s*==\s*['\"]__main__['\"]:\s*\n(?:(?:[ \t]+.*)?\n)*",
        "",
        content,
        flags=re.MULTILINE,
    )
    return re.sub(r"^mod\s*=\s*Repro\(\)\s*$", "", content, flags=re.MULTILINE)


def load_full_graph_definition(graph_path: str | Path) -> FullGraphDefinition:
    import torch
    import torch.fx as fx
    import torch.nn
    import torch._inductor.inductor_prims  # noqa: F401 - registers inductor prims.

    path = Path(graph_path)
    raw_content = path.read_text()
    content = _strip_runnable_boilerplate(raw_content)
    sidecar = load_full_graph_sidecar(path)
    annotation_inputs = parse_full_graph_inputs(raw_content)
    annotation_attrs = parse_full_graph_tensor_attrs(raw_content)
    _validate_sidecar_inputs(sidecar.get("inputs"), annotation_inputs)
    _validate_sidecar_tensor_attrs(sidecar.get("tensor_attrs"), annotation_attrs)

    exec_globals: dict[str, Any] = {"__builtins__": __builtins__}
    for attr in dir(torch.nn):
        if not attr.startswith("_"):
            exec_globals[attr] = getattr(torch.nn, attr)
    exec_globals.update({
        "torch": torch,
        "fx": fx,
        "inf": math.inf,
        "nan": float("nan"),
        "device": torch.device,
        "tensor": torch.tensor,
    })

    exec(compile(content, str(path), "exec"), exec_globals)

    graph_cls = None
    for name, value in exec_globals.items():
        if (
            isinstance(value, type)
            and issubclass(value, torch.nn.Module)
            and value is not torch.nn.Module
            and name not in dir(torch.nn)
        ):
            graph_cls = value
            break
    if graph_cls is None:
        raise RuntimeError(f"{path} does not define a torch.nn.Module graph class")

    return FullGraphDefinition(
        path=path,
        graph_cls=graph_cls,
        input_specs=sidecar.get("inputs") or annotation_inputs,
        tensor_attrs=sidecar.get("tensor_attrs") or annotation_attrs,
        forward_takes_no_inputs=forward_takes_no_inputs(raw_content),
        metadata={
            "source": sidecar.get("source") or infer_full_graph_source(path),
            "sidecar": sidecar,
        },
    )


def _resolve_device(requested: str | None, default_device: str):
    import torch

    target = requested or default_device
    if target.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda:0")
    if target.startswith("cpu") or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(default_device)


def _is_integer_dtype(dtype) -> bool:
    import torch

    integer_dtypes = [
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ]
    for name in ("uint16", "uint32", "uint64"):
        if hasattr(torch, name):
            integer_dtypes.append(getattr(torch, name))
    return dtype in tuple(integer_dtypes)


def _is_float8_dtype(dtype) -> bool:
    import torch

    return dtype in tuple(
        getattr(torch, name)
        for name in ("float8_e4m3fn", "float8_e5m2")
        if hasattr(torch, name)
    )


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(int(dim), 1)
    return tuple(reversed(stride))


def _effective_stride(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if stride is not None and len(stride) == len(shape):
        return tuple(int(dim) for dim in stride)
    return _contiguous_stride(shape)


def _dense_storage_size(
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    *,
    storage_offset: int = 0,
) -> int:
    storage_offset = max(int(storage_offset), 0)
    if not shape:
        return storage_offset + 1
    stride = _effective_stride(shape, stride)
    return max(
        storage_offset + sum(s * (d - 1) for s, d in zip(stride, shape) if d > 1) + 1,
        1,
    )


def _view_storage(
    storage,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
):
    if not shape:
        return storage.as_strided((), (), storage_offset=storage_offset)
    return storage.as_strided(
        shape,
        _effective_stride(shape, stride),
        storage_offset=storage_offset,
    )


def _generation_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    gen = spec.get("gen")
    if gen is not None:
        return gen
    gen = spec.get("generator")
    if gen is not None:
        return gen
    return None


def _make_permutation_tensor(
    shape: tuple[int, ...],
    dtype,
    device,
    *,
    stride: tuple[int, ...] | list[int] | None = None,
    storage_offset: int = 0,
    size: int | None = None,
):
    import torch

    logical_numel = _numel(shape)
    size = int(size if size is not None else logical_numel)
    if size == logical_numel:
        values = torch.randperm(size, dtype=torch.int64, device=device).reshape(shape)
    elif logical_numel % size == 0:
        rows = logical_numel // size
        values = torch.stack(
            [torch.randperm(size, dtype=torch.int64, device=device) for _ in range(rows)],
            dim=0,
        ).reshape(shape)
    else:
        raise ValueError(
            "permutation generator needs size == numel or size to divide numel, "
            f"got size={size}, numel={logical_numel}"
        )
    values = values.to(dtype)
    if stride is None and storage_offset == 0:
        return values
    storage = torch.empty(
        (_dense_storage_size(shape, stride, storage_offset=storage_offset),),
        dtype=dtype,
        device=device,
    )
    out = _view_storage(storage, shape, stride, storage_offset)
    out.copy_(values)
    return out


def _make_exact_tensor_from_spec(
    spec: dict[str, Any],
    *,
    dtype,
    device,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
):
    import torch

    if not spec.get("exact") or "data" not in spec:
        return None
    data = spec.get("data") or []
    source_dtype = torch.float32 if _is_float8_dtype(dtype) else dtype
    logical = torch.tensor(data, dtype=source_dtype, device=device).to(dtype)
    logical = logical.reshape(shape if shape else ())
    if stride is None and storage_offset == 0:
        return logical
    storage = torch.empty(
        (_dense_storage_size(shape, stride, storage_offset=storage_offset),),
        dtype=dtype,
        device=device,
    )
    out = _view_storage(storage, shape, stride, storage_offset)
    out.copy_(logical)
    return out


def _dtype_iinfo_max(dtype) -> int | None:
    """Return the max representable value for an integer dtype, or None for non-int."""
    import torch

    try:
        return torch.iinfo(dtype).max
    except (TypeError, RuntimeError):
        return None


def _randint_storage(
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
    dtype,
    device,
    low: int,
    high: int,
):
    import torch

    # Clamp high to dtype max+1 to avoid overflow (e.g. int8 max is 127)
    dtype_max = _dtype_iinfo_max(dtype)
    if dtype_max is not None and high > dtype_max + 1:
        high = dtype_max + 1
    if high <= low:
        high = low + 1
    storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
    storage = torch.randint(low, high, (storage_size,), dtype=dtype, device=device)
    return _view_storage(storage, shape, stride, storage_offset)


def _randn_storage(
    *,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | list[int] | None,
    storage_offset: int,
    dtype,
    device,
):
    import torch

    storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
    if _is_float8_dtype(dtype):
        storage = torch.randn((storage_size,), dtype=torch.float32, device=device).to(dtype)
    else:
        storage = torch.randn((storage_size,), dtype=dtype, device=device)
    return _view_storage(storage, shape, stride, storage_offset)


def make_tensor_from_spec(spec: dict[str, Any], *, default_device: str = "cuda"):
    import torch

    dtype = _torch_dtype(spec.get("dtype", "float32"))
    shape = tuple(spec.get("shape", ()))
    stride = spec.get("stride")
    dev = _resolve_device(spec.get("device"), default_device)
    storage_offset = int(spec.get("storage_offset", 0) or 0)
    generator = _generation_spec(spec) or _default_generator_for_dtype(
        _normalize_dtype_name(spec.get("dtype", "float32")),
        shape,
    )

    exact = _make_exact_tensor_from_spec(
        spec,
        dtype=dtype,
        device=dev,
        shape=shape,
        stride=stride,
        storage_offset=storage_offset,
    )
    if exact is not None:
        return exact
    if spec.get("requires_exact"):
        # Integer tensor constants without exact data: use zeros as a safe
        # fallback. Zero is always a valid index for indexing ops.
        import warnings

        name = spec.get("name", "<unnamed>")
        warnings.warn(
            f"Full graph tensor attribute {name!r} requires exact data but none "
            "available; using zeros as fallback. Results may not be numerically "
            "correct but the graph will be runnable for benchmarking.",
            stacklevel=2,
        )
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        base = torch.zeros((storage_size,), dtype=dtype, device=dev)
        return _view_storage(base, shape, stride, storage_offset)

    if generator.get("kind") == "constant":
        value = generator.get("value", 0)
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        base = torch.full((storage_size,), value, dtype=dtype, device=dev)
        return _view_storage(base, shape, stride, storage_offset)
    if generator.get("kind") == "permutation":
        return _make_permutation_tensor(
            shape,
            dtype,
            dev,
            stride=stride,
            storage_offset=storage_offset,
            size=generator.get("size"),
        )
    if generator.get("kind") == "index":
        low = int(generator.get("low", 0))
        high = int(generator.get("high", 100))
        if _is_integer_dtype(dtype):
            return _randint_storage(
                shape=shape,
                stride=stride,
                storage_offset=storage_offset,
                dtype=dtype,
                device=dev,
                low=low,
                high=high,
            )
        if high <= low:
            high = low + 1
        storage_size = _dense_storage_size(shape, stride, storage_offset=storage_offset)
        storage = torch.randint(
            low,
            high,
            (storage_size,),
            dtype=torch.int64,
            device=dev,
        ).to(dtype)
        return _view_storage(storage, shape, stride, storage_offset)
    elif dtype == torch.bool:
        return _randint_storage(
            shape=shape,
            stride=stride,
            storage_offset=storage_offset,
            dtype=torch.bool,
            device=dev,
            low=0,
            high=2,
        )
    elif _is_integer_dtype(dtype):
        high = max([int(dim) for dim in shape], default=100)
        return _randint_storage(
            shape=shape,
            stride=stride,
            storage_offset=storage_offset,
            dtype=dtype,
            device=dev,
            low=0,
            high=max(high, 1),
        )
    return _randn_storage(
        shape=shape,
        stride=stride,
        storage_offset=storage_offset,
        dtype=dtype,
        device=dev,
    )


def make_inputs_from_full_graph_specs(
    input_specs: list[dict[str, Any]],
    *,
    default_device: str = "cuda",
) -> tuple[Any, ...]:
    inputs = []
    for spec in input_specs:
        kind = spec.get("kind")
        if kind == "symint":
            inputs.append(int(spec.get("value", 32)))
        elif kind == "scalar":
            inputs.append(spec.get("value", 1))
        elif kind == "tensor":
            inputs.append(make_tensor_from_spec(spec, default_device=default_device))
        else:
            inputs.append(1)
    return tuple(inputs)


def instantiate_full_graph(
    definition: FullGraphDefinition,
    *,
    default_device: str = "cuda",
):
    instance = definition.graph_cls()
    for attr_name, spec in definition.tensor_attrs.items():
        if hasattr(instance, attr_name):
            continue
        instance.register_buffer(
            attr_name,
            make_tensor_from_spec(spec, default_device=default_device),
        )
    return instance


def load_full_graph(
    graph_path_or_definition: str | Path | FullGraphDefinition,
    *,
    default_device: str = "cuda",
):
    definition = (
        graph_path_or_definition
        if isinstance(graph_path_or_definition, FullGraphDefinition)
        else load_full_graph_definition(graph_path_or_definition)
    )
    if not definition.input_specs and not definition.forward_takes_no_inputs:
        raise RuntimeError(f"{definition.path} has no parseable forward inputs")
    instance = instantiate_full_graph(definition, default_device=default_device)
    inputs = make_inputs_from_full_graph_specs(
        definition.input_specs,
        default_device=default_device,
    )
    return instance, inputs, definition


def tensor_bytes(value: Any) -> int:
    import torch

    seen: set[tuple[int, int, int]] = set()

    def visit(obj: Any) -> int:
        if torch.is_tensor(obj):
            key = (
                obj.untyped_storage().data_ptr(),
                obj.untyped_storage().nbytes(),
                obj.storage_offset(),
            )
            if key in seen:
                return 0
            seen.add(key)
            return obj.nelement() * obj.element_size()
        if isinstance(obj, dict):
            return sum(visit(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(visit(v) for v in obj)
        return 0

    return visit(value)
