"""Backfill integer input constraints for full graph .meta.json sidecars.

Many full graphs have integer tensor inputs (e.g., token IDs, position IDs)
whose valid ranges depend on downstream consumers (embedding tables, gather ops).
Without proper constraints, the benchmark harness skips these graphs as unsafe.

This script:
1. Loads each full_graph_*.py that has unsafe integer inputs
2. Parses the forward() source to build a lightweight dataflow graph
3. Runs consumer-based bounds inference (embedding/gather/scatter/index)
4. Writes or updates the .meta.json sidecar with graph-inferred constraints

Usage:
    python scripts/backfill_integer_constraints.py [--dry-run] [--verbose]
    python scripts/backfill_integer_constraints.py --path repros/models/hf/infer/AlbertForMaskedLM/full_graph_000.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from full_graph_harness import (
    parse_full_graph_inputs,
    _DTYPE_SHORT_NAMES,
)


# ---------------------------------------------------------------------------
# Source-level graph analysis
# ---------------------------------------------------------------------------

def _parse_variable_shapes(content: str) -> dict[str, list[int]]:
    """Parse all annotated variable shapes from source: var: "dtype[d1, d2]" = ..."""
    var_shapes = {}
    for m in re.finditer(r'(\w+):\s*"(\w+)\[([^\]]*)\]', content):
        vname = m.group(1)
        dims_str = m.group(3).strip()
        if dims_str:
            try:
                shape = [int(x.strip()) for x in dims_str.split(",")]
                var_shapes[vname] = shape
            except ValueError:
                pass  # Skip symbolic shapes
        else:
            var_shapes[vname] = []
    return var_shapes


def _parse_variable_dtypes(content: str) -> dict[str, str]:
    """Parse all annotated variable dtypes from source: var: "dtype[...]" = ..."""
    var_dtypes = {}
    for m in re.finditer(r'(\w+):\s*"(\w+)\[', content):
        var_dtypes[m.group(1)] = m.group(2)
    return var_dtypes


# Op patterns we look for in the source code
_OP_CALL_RE = re.compile(
    r'(\w+)(?::\s*"[^"]*")?\s*=\s*'
    r'torch\.ops\.([\w.]+)\('
    r'([^)]*)\)'
)

# Also match: var = torch.ops.X(args);  trailing_kill
# Use [^;\n]* to avoid matching across newlines (each op is on one line)
_OP_CALL_RE2 = re.compile(
    r'(\w+)(?::\s*"[^"]*")?\s*=\s*'
    r'torch\.ops\.([\w.]+)\('
    r'([^;\n]*)\)'
)


def _parse_op_calls(content: str) -> list[tuple[str, str, list[str]]]:
    """Parse op calls from the forward body.

    Returns list of (result_var, op_name, [arg_strings]).
    """
    ops = []
    for m in _OP_CALL_RE2.finditer(content):
        result_var = m.group(1)
        op_name = m.group(2)
        args_str = m.group(3).strip()
        # Parse args (split by comma, but handle nested brackets)
        args = _split_args(args_str)
        ops.append((result_var, op_name, args))
    return ops


def _split_args(args_str: str) -> list[str]:
    """Split a comma-separated argument string, respecting brackets."""
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return [a for a in args if a]


def _arg_is_var(arg: str) -> str | None:
    """If arg is a simple variable name, return it. Otherwise None."""
    arg = arg.strip()
    if re.fullmatch(r'[a-zA-Z_]\w*', arg):
        return arg
    return None


def _arg_is_int(arg: str) -> int | None:
    """If arg is an integer literal, return it. Otherwise None."""
    arg = arg.strip()
    try:
        return int(arg)
    except ValueError:
        return None


def _arg_is_list_of_vars(arg: str) -> list[str | None] | None:
    """If arg looks like [var1, var2, ...], return list of var names (None for non-vars)."""
    arg = arg.strip()
    if arg.startswith("[") and arg.endswith("]"):
        inner = arg[1:-1]
        items = [x.strip() for x in inner.split(",")]
        result = []
        for item in items:
            v = _arg_is_var(item)
            result.append(v)
        return result
    return None


# Passthrough ops: these don't change the "index identity" of their input
PASSTHROUGH_OPS = {
    "aten.clone.default",
    "aten.reshape.default",
    "aten.view.default",
    "aten.expand.default",
    "aten.slice.Tensor",
    "aten.unsqueeze.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.select.int",
    "aten.permute.default",
    "aten.contiguous.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.lift_fresh_copy.default",
    "prims.convert_element_type.default",
    "aten._to_copy.default",
    "aten.constant_pad_nd.default",
    "aten.cat.default",
}


def infer_bounds_from_source(content: str) -> tuple[dict[str, int], dict[str, int]]:
    """Infer index bounds and permutation sizes from forward() source code.

    Returns (index_bounds, perm_bounds) mapping input_name -> bound/size.
    """
    var_shapes = _parse_variable_shapes(content)
    var_dtypes = _parse_variable_dtypes(content)

    # Parse input annotations
    specs = parse_full_graph_inputs(content)
    int_input_names = set()
    for spec in specs:
        if spec.get("kind") != "tensor":
            continue
        dtype = spec.get("dtype", "")
        if dtype.startswith("int") or dtype.startswith("uint"):
            int_input_names.add(spec["name"])

    if not int_input_names:
        return {}, {}

    # Build a "derives from" map using source-level dataflow
    # derives_from[var] = set of input names that this var derives from
    derives_from: dict[str, set[str]] = {}
    for name in int_input_names:
        derives_from[name] = {name}

    # Parse all op calls from the forward body
    op_calls = _parse_op_calls(content)

    for result_var, op_name, args in op_calls:
        if op_name in PASSTHROUGH_OPS:
            # First arg is the source
            if args:
                src = _arg_is_var(args[0])
                if src and src in derives_from:
                    derives_from[result_var] = set(derives_from[src])

        elif op_name == "aten.where.self":
            # where(cond, x, y) -> result derives from x and y
            sources = set()
            for arg_str in args[1:3]:  # args[1] and args[2]
                v = _arg_is_var(arg_str)
                if v and v in derives_from:
                    sources.update(derives_from[v])
            if sources:
                derives_from[result_var] = sources

        elif op_name in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor"):
            # Arithmetic: trace through the node operand
            if len(args) >= 2:
                v0 = _arg_is_var(args[0])
                v1 = _arg_is_var(args[1])
                if v0 and v0 in derives_from:
                    derives_from[result_var] = set(derives_from[v0])
                elif v1 and v1 in derives_from:
                    derives_from[result_var] = set(derives_from[v1])

        elif op_name == "aten.gather.default":
            # gather(input, dim, index): OUTPUT carries values from input
            if len(args) >= 1:
                src = _arg_is_var(args[0])
                if src and src in derives_from:
                    derives_from[result_var] = set(derives_from[src])

    # Now scan for index consumer ops
    index_bounds: dict[str, int] = {}  # input_name -> bound
    perm_bounds: dict[str, int] = {}

    def _set_bound(index_var: str, bound: int):
        """Set bound for all integer inputs that derive into index_var."""
        sources = derives_from.get(index_var, set())
        int_sources = sources & int_input_names
        for name in int_sources:
            if name not in index_bounds or index_bounds[name] > bound:
                index_bounds[name] = bound

    for result_var, op_name, args in op_calls:
        if op_name == "aten.embedding.default":
            # embedding(weight, indices, ...)
            if len(args) >= 2:
                weight_var = _arg_is_var(args[0])
                indices_var = _arg_is_var(args[1])
                if indices_var and weight_var:
                    # Check if indices derives from an integer input
                    sources = derives_from.get(indices_var, set())
                    if sources & int_input_names:
                        weight_shape = var_shapes.get(weight_var, [])
                        if weight_shape:
                            _set_bound(indices_var, weight_shape[0])

        elif op_name == "aten.gather.default":
            # gather(input, dim, index)
            if len(args) >= 3:
                input_var = _arg_is_var(args[0])
                dim = _arg_is_int(args[1])
                index_var = _arg_is_var(args[2])
                if index_var and input_var and dim is not None:
                    sources = derives_from.get(index_var, set())
                    if sources & int_input_names:
                        input_shape = var_shapes.get(input_var, [])
                        if input_shape and dim < len(input_shape):
                            _set_bound(index_var, input_shape[dim])

        elif op_name in ("aten.scatter.src", "aten.scatter.value"):
            # scatter.src(target, dim, index, src)
            # scatter.value(target, dim, index, value)
            if len(args) >= 3:
                target_var = _arg_is_var(args[0])
                dim = _arg_is_int(args[1])
                index_var = _arg_is_var(args[2])
                if index_var and target_var and dim is not None:
                    sources = derives_from.get(index_var, set())
                    if sources & int_input_names:
                        target_shape = var_shapes.get(target_var, [])
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_var, target_shape[actual_dim])

        elif op_name == "aten.scatter_add.default":
            # scatter_add(target, dim, index, src)
            if len(args) >= 3:
                target_var = _arg_is_var(args[0])
                dim = _arg_is_int(args[1])
                index_var = _arg_is_var(args[2])
                if index_var and target_var and dim is not None:
                    sources = derives_from.get(index_var, set())
                    if sources & int_input_names:
                        target_shape = var_shapes.get(target_var, [])
                        if target_shape:
                            actual_dim = dim if dim >= 0 else dim + len(target_shape)
                            if 0 <= actual_dim < len(target_shape):
                                _set_bound(index_var, target_shape[actual_dim])

        elif op_name == "aten.index_put.default":
            # index_put(target, [indices...], values, accumulate?)
            if len(args) >= 2:
                target_var = _arg_is_var(args[0])
                indices_list = _arg_is_list_of_vars(args[1])
                if target_var and indices_list:
                    target_shape = var_shapes.get(target_var, [])
                    if target_shape:
                        for dim_idx, idx_var in enumerate(indices_list):
                            if idx_var and dim_idx < len(target_shape):
                                sources = derives_from.get(idx_var, set())
                                if sources & int_input_names:
                                    _set_bound(idx_var, target_shape[dim_idx])

        elif op_name == "aten.index.Tensor":
            # index.Tensor(input, [indices...])
            if len(args) >= 2:
                input_var = _arg_is_var(args[0])
                indices_list = _arg_is_list_of_vars(args[1])
                if input_var and indices_list:
                    input_shape = var_shapes.get(input_var, [])
                    if input_shape:
                        for dim_idx, idx_var in enumerate(indices_list):
                            if idx_var and dim_idx < len(input_shape):
                                sources = derives_from.get(idx_var, set())
                                if sources & int_input_names:
                                    _set_bound(idx_var, input_shape[dim_idx])

        elif op_name == "aten.index_select.default":
            # index_select(input, dim, index)
            if len(args) >= 3:
                input_var = _arg_is_var(args[0])
                dim = _arg_is_int(args[1])
                index_var = _arg_is_var(args[2])
                if index_var and input_var and dim is not None:
                    sources = derives_from.get(index_var, set())
                    if sources & int_input_names:
                        input_shape = var_shapes.get(input_var, [])
                        if input_shape:
                            actual_dim = dim if dim >= 0 else dim + len(input_shape)
                            if 0 <= actual_dim < len(input_shape):
                                _set_bound(index_var, input_shape[actual_dim])

        elif op_name == "aten.one_hot.default":
            # one_hot(input, num_classes)
            if len(args) >= 2:
                input_var = _arg_is_var(args[0])
                num_classes = _arg_is_int(args[1])
                if input_var and num_classes is not None:
                    sources = derives_from.get(input_var, set())
                    if sources & int_input_names:
                        _set_bound(input_var, num_classes)

        elif op_name == "aten._embedding_bag.default":
            # _embedding_bag(weight, indices, offsets, ...)
            if len(args) >= 2:
                weight_var = _arg_is_var(args[0])
                indices_var = _arg_is_var(args[1])
                if indices_var and weight_var:
                    sources = derives_from.get(indices_var, set())
                    if sources & int_input_names:
                        weight_shape = var_shapes.get(weight_var, [])
                        if weight_shape:
                            _set_bound(indices_var, weight_shape[0])

    return index_bounds, perm_bounds


# ---------------------------------------------------------------------------
# Sidecar update logic
# ---------------------------------------------------------------------------

def _is_integer_non_bool_dtype(dtype_name: str) -> bool:
    return (
        dtype_name.startswith("int") or dtype_name.startswith("uint")
    ) and dtype_name != "bool"


def _has_unsafe_integer_inputs(specs: list[dict], sidecar: dict | None) -> bool:
    """Check if any integer tensor input lacks graph-inferred constraints."""
    sidecar_inputs = (sidecar or {}).get("inputs", []) if sidecar else []
    check_inputs = sidecar_inputs if sidecar_inputs else specs

    for spec in check_inputs:
        if spec.get("kind") != "tensor":
            continue
        dtype = spec.get("dtype", "")
        if not _is_integer_non_bool_dtype(dtype):
            continue
        if spec.get("exact") and "data" in spec:
            continue
        gen = spec.get("gen") or spec.get("generator") or {}
        if gen.get("kind") in ("index", "permutation") and spec.get("constraint_source") == "graph_inference":
            continue
        return True
    return False


def update_sidecar_with_constraints(
    graph_path: Path,
    index_bounds: dict[str, int],
    perm_bounds: dict[str, int],
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Update the .meta.json sidecar with inferred constraints.

    Returns (updated_count, already_ok_count).
    """
    meta_path = graph_path.with_suffix(".meta.json")
    content = graph_path.read_text()
    specs = parse_full_graph_inputs(content)

    if meta_path.exists():
        sidecar = json.loads(meta_path.read_text())
    else:
        # Create a minimal sidecar from annotations
        sidecar = {
            "schema_version": 1,
            "graph": graph_path.name,
            "inputs": [],
        }

    # Get or initialize the inputs list
    if "inputs" not in sidecar or not sidecar["inputs"]:
        sidecar_inputs = []
        for spec in specs:
            sidecar_inputs.append(dict(spec))
        sidecar["inputs"] = sidecar_inputs

    updated_count = 0
    already_ok = 0

    for i, spec in enumerate(sidecar["inputs"]):
        if spec.get("kind") != "tensor":
            continue
        name = spec.get("name", f"arg{i}")
        dtype = spec.get("dtype", "")
        if not _is_integer_non_bool_dtype(dtype):
            continue

        # Already has graph-inferred constraints?
        gen = spec.get("gen") or spec.get("generator") or {}
        if gen.get("kind") in ("index", "permutation") and spec.get("constraint_source") == "graph_inference":
            already_ok += 1
            continue

        # Check if we have a constraint for this input
        if name in perm_bounds:
            new_gen = {"kind": "permutation", "size": int(perm_bounds[name])}
            spec["gen"] = new_gen
            spec["generator"] = new_gen
            spec["constraint_source"] = "graph_inference"
            updated_count += 1
        elif name in index_bounds:
            new_gen = {"kind": "index", "low": 0, "high": int(index_bounds[name])}
            spec["gen"] = new_gen
            spec["generator"] = new_gen
            spec["constraint_source"] = "graph_inference"
            updated_count += 1

    if updated_count > 0 and not dry_run:
        meta_path.write_text(json.dumps(sidecar, indent=2) + "\n")

    return updated_count, already_ok


def find_unsafe_full_graphs(root: Path = None) -> list[Path]:
    """Find all full_graph_*.py files with unsafe integer inputs."""
    if root is None:
        root = PROJECT_ROOT / "repros" / "models"

    unsafe = []
    for graph_path in sorted(root.rglob("full_graph_*.py")):
        content = graph_path.read_text()
        specs = parse_full_graph_inputs(content)

        meta_path = graph_path.with_suffix(".meta.json")
        sidecar = json.loads(meta_path.read_text()) if meta_path.exists() else None

        if _has_unsafe_integer_inputs(specs, sidecar):
            unsafe.append(graph_path)

    return unsafe


def main():
    parser = argparse.ArgumentParser(description="Backfill integer constraints for full graphs")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print details per graph")
    parser.add_argument("--path", type=str, help="Process a single graph file")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N graphs (0=all)")
    args = parser.parse_args()

    if args.path:
        graphs = [Path(args.path)]
    else:
        print("Finding full graphs with unsafe integer inputs...")
        graphs = find_unsafe_full_graphs()
        print(f"Found {len(graphs)} graphs to process")

    if args.limit > 0:
        graphs = graphs[:args.limit]

    total_updated = 0
    total_already_ok = 0
    total_no_bounds = 0
    graphs_fixed = 0

    for i, graph_path in enumerate(graphs):
        rel_path = graph_path.relative_to(PROJECT_ROOT) if graph_path.is_relative_to(PROJECT_ROOT) else graph_path

        content = graph_path.read_text()
        index_bounds, perm_bounds = infer_bounds_from_source(content)

        if not index_bounds and not perm_bounds:
            total_no_bounds += 1
            if args.verbose:
                print(f"  [{i+1}/{len(graphs)}] NO BOUNDS {rel_path}")
            continue

        updated, already_ok = update_sidecar_with_constraints(
            graph_path, index_bounds, perm_bounds, dry_run=args.dry_run
        )
        total_updated += updated
        total_already_ok += already_ok
        if updated > 0:
            graphs_fixed += 1

        if args.verbose and updated > 0:
            all_bounds = {**{k: f"Index({v})" for k, v in index_bounds.items()},
                         **{k: f"Perm({v})" for k, v in perm_bounds.items()}}
            print(f"  [{i+1}/{len(graphs)}] UPDATED {rel_path}: {updated} inputs: {all_bounds}")

        if (i + 1) % 50 == 0 and not args.verbose:
            print(f"  Processed {i+1}/{len(graphs)}...")

    print(f"\nResults:")
    print(f"  Graphs processed: {len(graphs)}")
    print(f"  Graphs fixed: {graphs_fixed}")
    print(f"  Inputs updated: {total_updated}")
    print(f"  Inputs already OK: {total_already_ok}")
    print(f"  Graphs with no inferable bounds: {total_no_bounds}")
    if args.dry_run:
        print(f"\n  (DRY RUN - no files written)")
    else:
        remaining = len(graphs) - graphs_fixed - total_no_bounds
        print(f"  Graphs still unsafe (partial fix): {remaining}")


if __name__ == "__main__":
    main()
