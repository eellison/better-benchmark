"""Replay dependencies of a captured graph, recorded at export time.

A saved graph can only replay in a process where every operator it calls is
registered. ``replay_dependencies`` reads that off the live GraphModule while
it is being exported (custom operators, including ones passed as arguments to
higher-order operators, collectives with their group size, and user-defined
Triton kernels), and
``check_replay_dependencies`` asks the current process whether each one is
available, so a graph that cannot run here is reported with the reason instead
of failing partway through loading or hanging on a collective.

Nothing here reads graph source text.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

SCHEMA_VERSION = 1

BUILTIN_OP_NAMESPACES = frozenset({
    "aten",
    "prims",
    "prim",
    "rngprims",
    "debugprims",
    "higher_order",
    "inductor",
    "quantized",
    "_quantized",
    "quantized_decomposed",
    "profiler",
    "mkldnn",
    "mkldnn_prepacked",
    "mkl",
    "onednn",
    "streams",
    "sparse",
    "quantization",
    "inductor_prims",
    "_dtensor",
    "fsdp",
    "export",
    "onnx",
    "_native",
    "mempool",
    "debug_mode_ops",
    "static_runtime",
    "_test",
    "_inductor_test",
})
COLLECTIVE_OP_NAMESPACES = frozenset({
    "_c10d_functional",
    "c10d_functional",
    "_c10d_functional_autograd",
    "c10d",
    "symm_mem",
})
_NON_COLLECTIVE_OPS = frozenset({"wait_tensor"})
# A saved graph names a user-defined Triton kernel only by its index into the
# capturing process's kernel_side_table, so no other process can resolve it:
# a fresh one fails, and one holding other kernels silently runs those.
USER_TRITON_HOPS = frozenset({
    "triton_kernel_wrapper_functional",
    "triton_kernel_wrapper_mutation",
})


def _qualified_name(op: Any) -> str | None:
    """``ns::name.overload`` for an OpOverload, ``ns::name`` for a packet."""
    import torch

    if isinstance(op, torch._ops.OpOverload):
        return f"{op._schema.name}.{op._overloadname}"
    if isinstance(op, torch._ops.OpOverloadPacket):
        return op._qualified_op_name
    return None


def _namespace(qualified: str) -> str:
    return qualified.split("::", 1)[0]


def _leaves(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _user_triton_kernel_name(node: Any) -> str | None:
    """Name of the kernel at ``node``'s side-table index, when this process's
    table holds a kernel there whose parameters accept the node's arguments.

    The table is process-global, so in a process other than the capturing one
    the index can hold an unrelated kernel; the parameter check rejects most
    of those, and the name is only informational."""
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

    idx = node.kwargs.get("kernel_idx")
    if idx not in kernel_side_table.id_to_kernel:
        return None
    fn = kernel_side_table.get_kernel(idx)
    # Autotuner / Heuristics wrap the JITFunction in .fn.
    while not hasattr(fn, "arg_names") and hasattr(fn, "fn"):
        fn = fn.fn
    arg_names = getattr(fn, "arg_names", None)
    if arg_names is None or not set(node.kwargs.get("kwargs") or {}) <= set(arg_names):
        return None
    return getattr(fn, "__name__", None)


def user_triton_hops_in_class(graph_cls: type) -> list[str]:
    """User-Triton HOPs a loaded graph class calls, from the bytecode names of
    its methods and of the nested module classes that hold HOP subgraphs
    (cond / while_loop / invoke_subgraph bodies).

    For sidecars written before ``replay_dependencies`` recorded the kernels."""
    found: set[str] = set()
    pending, seen = [graph_cls], set()
    while pending:
        cls = pending.pop()
        if cls in seen:
            continue
        seen.add(cls)
        for value in vars(cls).values():
            if isinstance(value, type):
                pending.append(value)
                continue
            code = getattr(value, "__code__", None)
            found.update(USER_TRITON_HOPS.intersection(getattr(code, "co_names", ())))
    return sorted(found)


def full_graph_calls_user_triton(graph_cls: type, sidecar: dict | None) -> bool:
    """Whether a saved full graph launches user-defined Triton kernels,
    according to either its code or its sidecar."""
    if user_triton_hops_in_class(graph_cls):
        return True
    sidecar = sidecar or {}
    if "user_triton" in (sidecar.get("masked") or {}).get("reasons", ()):
        return True
    return any(
        item["category"] == "user_triton_kernel"
        for item in check_replay_dependencies(sidecar)["blocked"]
    )


def _graph_modules(gm: Any):
    import torch.fx as fx

    for module in gm.modules():
        if isinstance(module, fx.GraphModule):
            yield module


def _module_for_file(path: str) -> str | None:
    target = os.path.realpath(path)
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if module_file and os.path.realpath(module_file) == target:
            return None if name == "__main__" else name
    return None


def _registration_file(op_name: str) -> str | None:
    import torch

    try:
        dump = torch._C._dispatch_dump(op_name)
    except Exception:
        return None
    for line in dump.splitlines():
        if line.startswith("debug: registered at "):
            location = line[len("debug: registered at "):].strip()
            path, sep, lineno = location.rpartition(":")
            return path if sep and lineno.isdigit() else location
    return None


def _python_provider(op_name: str) -> str | None:
    from torch._library import custom_ops

    opdef = custom_ops.OPDEFS.get(op_name)
    if opdef is not None:
        module = getattr(opdef._init_fn, "__module__", None)
    else:
        path = _registration_file(op_name)
        module = _module_for_file(path) if path else None
    if module is None or module == "torch" or module.startswith("torch."):
        return None
    return module


def _registered_from_python(op_name: str) -> bool:
    from torch._library import custom_ops

    path = _registration_file(op_name)
    return op_name in custom_ops.OPDEFS or bool(path and path.endswith(".py"))


def _loaded_library_names() -> list[str]:
    import torch

    return sorted({os.path.basename(p) for p in torch.ops.loaded_libraries})


def _group(node: Any, op: Any) -> tuple[str | None, int | None]:
    """``(group_name, group_size)`` of a collective node, where known."""
    import torch

    schema_args = [a.name for a in op._schema.arguments]

    def arg(name: str):
        if name in node.kwargs:
            return node.kwargs[name]
        if name in schema_args:
            idx = schema_args.index(name)
            if idx < len(node.args):
                return node.args[idx]
        return None

    name = arg("group_name")
    name = name if isinstance(name, str) else None
    size = arg("group_size")
    if isinstance(size, int):
        return name, size
    return name, _resolved_group_size(name) if name else None


def _resolved_group_size(name: str) -> int | None:
    """Size of the process group ``name`` in this process, or None if it doesn't exist."""
    import torch

    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return None
    try:
        from torch.distributed.distributed_c10d import _resolve_process_group

        return _resolve_process_group(name).size()
    except Exception:
        return None


def replay_dependencies(gm: Any) -> dict[str, Any]:
    """What ``gm`` needs from the replaying process beyond stock torch."""
    import torch

    custom: set[str] = set()
    collectives: set[str] = set()
    group_sizes: set[int] = set()
    groups: dict[str, int | None] = {}
    user_triton: dict[tuple, int] = {}
    for module in _graph_modules(gm):
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue
            target = node.target
            if (
                isinstance(target, torch._ops.HigherOrderOperator)
                and target.name() in USER_TRITON_HOPS
            ):
                key = (node.kwargs.get("kernel_idx"), _user_triton_kernel_name(node))
                user_triton[key] = user_triton.get(key, 0) + 1
                continue
            if isinstance(target, torch._ops.HigherOrderOperator):
                for leaf in _leaves((node.args, node.kwargs)):
                    name = _qualified_name(leaf)
                    if name and _namespace(name) not in (
                        BUILTIN_OP_NAMESPACES | COLLECTIVE_OP_NAMESPACES
                    ):
                        custom.add(name)
                continue
            name = _qualified_name(target)
            if name is None:
                continue
            namespace = _namespace(name)
            if namespace in COLLECTIVE_OP_NAMESPACES:
                if target._schema.name.split("::", 1)[1] in _NON_COLLECTIVE_OPS:
                    continue
                collectives.add(name)
                group_name, size = _group(node, target)
                if size is not None:
                    group_sizes.add(size)
                if group_name is not None:
                    groups[group_name] = size
            elif namespace not in BUILTIN_OP_NAMESPACES:
                custom.add(name)

    deps: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if custom:
        libraries = _loaded_library_names()
        entries = []
        for name in sorted(custom):
            op_name = name.split(".", 1)[0]
            provider: dict[str, Any] = {"python_module": _python_provider(op_name)}
            if (
                provider["python_module"] is None
                and libraries
                and not _registered_from_python(op_name)
            ):
                provider["libraries"] = libraries
            entries.append({"op": name, "provider": provider})
        deps["custom_ops"] = entries
    if collectives:
        deps["distributed"] = {
            "collectives": sorted(collectives),
            "group_size": max(group_sizes) if group_sizes else None,
        }
        if groups:
            deps["distributed"]["groups"] = [
                {"name": name, "size": size} for name, size in sorted(groups.items())
            ]
    if user_triton:
        deps["user_triton_kernels"] = [
            {"kernel_idx": idx, "kernel": name, "calls": calls}
            for (idx, name), calls in sorted(
                user_triton.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
            )
        ]
    return deps


def op_available(qualified: str) -> bool:
    """Whether ``ns::name[.overload]`` is registered in this process."""
    import torch

    namespace, _, rest = qualified.partition("::")
    name, _, overload = rest.partition(".")
    try:
        packet = getattr(getattr(torch.ops, namespace), name)
        return not overload or overload in packet.overloads()
    except (AttributeError, RuntimeError):
        return False


def _op_hint(provider: dict[str, Any]) -> str:
    options = []
    if provider.get("python_module"):
        options.append(f"import {provider['python_module']}")
    if provider.get("libraries"):
        options.append(
            "load the extension that registers it (loaded at capture: "
            + ", ".join(provider["libraries"]) + ")"
        )
    if not options:
        return "Register the operator (import or load its provider) before replay."
    return " or ".join(options) + " before replay."


def check_replay_dependencies(
    meta_or_deps: dict[str, Any] | None,
    *,
    import_providers: bool = False,
) -> dict[str, Any]:
    """Return {"status": "ready" | "blocked" | "unknown", "blocked": [...]}.

    Accepts a full-graph sidecar or its ``replay_dependencies`` block.
    "unknown" means the graph was captured without the block.
    """
    import torch

    deps = meta_or_deps or {}
    if "replay_dependencies" in deps:
        deps = deps["replay_dependencies"] or {}
        if not deps:
            return {"status": "unknown", "blocked": []}
    elif not deps or not set(deps) <= {
        "schema_version", "custom_ops", "distributed", "user_triton_kernels",
    }:
        return {"status": "unknown", "blocked": []}

    blocked = []
    for entry in deps.get("custom_ops") or []:
        provider = entry.get("provider") or {}
        module = provider.get("python_module")
        if import_providers and module and not op_available(entry["op"]):
            try:
                importlib.import_module(module)
            except Exception:
                pass
        if not op_available(entry["op"]):
            blocked.append({
                "category": "missing_custom_op",
                "item": entry["op"],
                "reason": f"custom op {entry['op']} is not registered",
                "hint": _op_hint(provider),
            })

    distributed = deps.get("distributed")
    if distributed and distributed.get("collectives"):
        size = distributed.get("group_size")
        dist = torch.distributed
        initialized = dist.is_available() and dist.is_initialized()
        item = ", ".join(distributed["collectives"])
        if not initialized or (size and dist.get_world_size() < size):
            need = f"a process group of size {size}" if size else "an initialized process group"
            blocked.append({
                "category": "distributed",
                "item": item,
                "reason": f"graph uses collectives and needs {need}",
                "hint": f"Run under torch.distributed with {need}.",
            })
        else:
            for group in distributed.get("groups") or []:
                found = _resolved_group_size(group["name"])
                if found is None or (group["size"] and found != group["size"]):
                    need = f"process group {group['name']!r}"
                    if group["size"]:
                        need += f" of size {group['size']}"
                    blocked.append({
                        "category": "distributed",
                        "item": item,
                        "reason": f"graph uses collectives on {need}, which this process does not have",
                        "hint": f"Create {need} before replay.",
                    })

    for entry in deps.get("user_triton_kernels") or []:
        blocked.append(user_triton_blocker(entry.get("kernel"), entry.get("kernel_idx")))

    return {"status": "blocked" if blocked else "ready", "blocked": blocked}


def user_triton_blocker(kernel: str | None, kernel_idx: int | None) -> dict[str, Any]:
    item = kernel or f"kernel_side_table index {kernel_idx}"
    return {
        "category": "user_triton_kernel",
        "item": item,
        "reason": (
            f"graph launches user-defined Triton kernel {item} by its index into "
            "the capturing process's kernel_side_table, which no other process "
            "can resolve"
        ),
        "hint": "Benchmark the graph's partitions; the kernel itself is not replayed.",
    }
