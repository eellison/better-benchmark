"""
Ingest post-grad FX graphs from CI tlparse artifacts.

Downloads tlparse artifacts from a GitHub Actions run, loads the
inductor_post_grad_graph_*.txt files as GraphModules, runs the capture
hook's partitioning on each, and merges the resulting regions into the
canonical repro set.

Usage:
    # Download and ingest from a specific run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --suite torchbench

    # Ingest from already-downloaded tlparse dir:
    python scripts/ingest_tlparse.py --tlparse-dir /tmp/tb_tlparse3 --suite torchbench

    # Just list what's available in a run:
    python scripts/ingest_tlparse.py --run-id 25956360525 --list-only
"""
import argparse
import ast
import functools
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def download_tlparse(run_id: int, suite: str | None, output_dir: Path) -> Path:
    """Download tlparse artifacts from a GitHub Actions run."""
    pattern = "tlparse-*"
    if suite:
        pattern = f"tlparse-*{suite}*"

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gh", "run", "download", str(run_id),
        "--repo", "pytorch/pytorch",
        "--pattern", pattern,
        "--dir", str(output_dir),
    ]
    print(f"Downloading: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return output_dir


def find_post_grad_graphs(tlparse_dir: Path) -> list[Path]:
    """Find all inductor_post_grad_graph_*.txt files."""
    return sorted(tlparse_dir.rglob("inductor_post_grad_graph_*.txt"))


def find_fx_graph_runnables(tlparse_dir: Path) -> list[Path]:
    """Find all fx_graph_runnable_*.txt files."""
    return sorted(tlparse_dir.rglob("fx_graph_runnable_*.txt"))


def find_matching_post_grad_graph(runnable_path: Path) -> Path | None:
    """Return the next post-grad trace emitted after a runnable."""
    match = re.fullmatch(r"fx_graph_runnable_(\d+)\.txt", runnable_path.name)
    if match is None:
        return None
    event_id = int(match.group(1))
    later_runnables = []
    for path in runnable_path.parent.glob("fx_graph_runnable_*.txt"):
        runnable_match = re.fullmatch(r"fx_graph_runnable_(\d+)\.txt", path.name)
        if runnable_match is not None and int(runnable_match.group(1)) > event_id:
            later_runnables.append(int(runnable_match.group(1)))
    next_runnable = min(later_runnables) if later_runnables else None
    candidates = []
    for candidate in runnable_path.parent.glob("inductor_post_grad_graph_*.txt"):
        candidate_match = re.fullmatch(
            r"inductor_post_grad_graph_(\d+)\.txt", candidate.name
        )
        if candidate_match is not None and int(candidate_match.group(1)) > event_id:
            candidates.append((int(candidate_match.group(1)), candidate))
    if not candidates:
        return None
    candidate_id, candidate = min(candidates)
    return candidate if next_runnable is None or candidate_id < next_runnable else None


def _evaluate_int_expression(expr: str, bindings: dict[str, int]) -> int:
    from fractions import Fraction

    def evaluate(node):
        if isinstance(node, ast.Name):
            if node.id not in bindings:
                raise ValueError(f"unbound symbol {node.id!r} in {expr!r}")
            return bindings[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return Fraction(left, right)
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        if isinstance(node, ast.Call):
            values = [evaluate(arg) for arg in node.args]
            if isinstance(node.func, ast.Name):
                if node.func.id == "FloorDiv" and len(values) == 2:
                    return values[0] // values[1]
                if node.func.id == "IntTrueDiv" and len(values) == 2:
                    return Fraction(values[0]) / Fraction(values[1])
                if node.func.id == "CeilToInt" and len(values) == 1:
                    return math.ceil(values[0])
            if isinstance(node.func, ast.Name) and node.func.id in {"Max", "max"}:
                return max(values)
            if isinstance(node.func, ast.Name) and node.func.id in {"Min", "min"}:
                return min(values)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in {"ceil", "floor"}
            ):
                return getattr(math, node.func.attr)(*values)
        raise ValueError(f"unsupported InputWriter expression: {ast.unparse(node)!r}")

    try:
        value = evaluate(ast.parse(expr, mode="eval").body)
    except ZeroDivisionError:
        raise ValueError(f"zero divisor in InputWriter expression: {expr!r}") from None
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    if not isinstance(value, int):
        raise ValueError(f"InputWriter expression is not integral: {expr!r}")
    return value


def infer_model_name(graph_path: Path) -> str:
    """Infer model name from the tlparse directory structure."""
    # Path looks like: tlparse-...-test-inductor_torchbench_perf-4-6-.../  -_0_0_0/  file.txt
    parts = str(graph_path).split("/")
    for part in parts:
        if "tlparse-" in part:
            # Extract suite and model info from directory name
            # e.g. tlparse-runattempt1-test-inductor_torchbench_perf_cuda_h100-1-9-...
            return part.split("_76")[0].replace("tlparse-runattempt1-test-", "")
    return "unknown"


def _parse_input_shapes_from_reader(content: str):
    """Parse concrete input specs from reader calls in load_args.

    Lines look like:
        reader.tensor(buf0, (32, 3, 3, 3), is_leaf=True)  # primals_1
        reader.tensor(buf2, (), dtype=torch.int64, is_leaf=True)  # primals_3
    """
    import torch

    tree = ast.parse(content)
    lines = content.splitlines()
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "reader"
            and node.func.attr
            in {"const", "generator", "opaque", "symint", "tensor", "unsupported"}
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )

    def keyword(call: ast.Call, name: str):
        return next((item.value for item in call.keywords if item.arg == name), None)

    def input_name(call: ast.Call) -> str | None:
        comment = lines[call.end_lineno - 1].partition("#")[2].strip()
        match = re.match(r"\w+", comment)
        return match.group(0) if match else None

    def reader_call(node: ast.AST, name: str) -> ast.Call | None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "reader":
            return None
        return node if node.func.attr == name else None

    def parse_storage_device(node: ast.AST) -> str:
        if not isinstance(node, ast.Call) or ast.unparse(node.func) not in {
            "device",
            "torch.device",
        }:
            raise ValueError(f"unsupported reader storage device: {ast.unparse(node)}")
        device_type_node = next(
            (item.value for item in node.keywords if item.arg == "type"),
            node.args[0] if node.args else None,
        )
        index_node = next(
            (item.value for item in node.keywords if item.arg == "index"),
            node.args[1] if len(node.args) > 1 else None,
        )
        device_type = ast.literal_eval(device_type_node)
        index = ast.literal_eval(index_node) if index_node is not None else None
        if not isinstance(device_type, str) or not isinstance(index, (int, type(None))):
            raise ValueError(f"unsupported reader storage device: {ast.unparse(node)}")
        return f"{device_type}:{index}" if index is not None else device_type

    bindings = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                bindings[node.targets[0].id] = value

    storage_devices = {}
    load_args_nodes = (
        node
        for statement in tree.body
        if isinstance(statement, ast.FunctionDef) and statement.name == "load_args"
        for node in ast.walk(statement)
    )
    for node in load_args_nodes:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue
        storage_call = reader_call(node.value, "storage")
        if storage_call is None:
            continue
        device_node = keyword(storage_call, "device")
        if device_node is not None:
            storage_devices[node.targets[0].id] = parse_storage_device(device_node)
    for call in calls:
        if call.func.attr != "symint" or not call.args:
            continue
        expr_node = keyword(call, "expr")
        if isinstance(expr_node, ast.Constant) and isinstance(expr_node.value, str):
            value = ast.literal_eval(call.args[0])
            if value is not None:
                bindings[expr_node.value] = int(value)

    def parse_int(value: str) -> int:
        return _evaluate_int_expression(value.strip(), bindings)

    def parse_dims(node: ast.AST) -> tuple[int, ...]:
        if not isinstance(node, (ast.List, ast.Tuple)):
            raise ValueError(f"reader dimensions must be a sequence: {ast.unparse(node)}")
        return tuple(parse_int(ast.unparse(dim)) for dim in node.elts)

    inputs = []
    for call in calls:
        if call.func.attr == "symint":
            if not call.args:
                raise ValueError("reader.symint requires a value argument")
            raw_value = ast.literal_eval(call.args[0])
            value = int(raw_value) if raw_value is not None else None
            expr_node = keyword(call, "expr")
            expr = (
                expr_node.value
                if isinstance(expr_node, ast.Constant)
                and isinstance(expr_node.value, str)
                else None
            )
            spec = {"kind": "symint", "name": input_name(call), "value": value}
            if expr is not None:
                spec["symbolic_expr"] = expr
            inputs.append(spec)
            continue

        if call.func.attr == "generator":
            if len(call.args) < 2:
                raise ValueError("reader.generator requires device type and index")
            inputs.append({
                "kind": "generator",
                "name": input_name(call),
                "device_type": str(ast.literal_eval(call.args[0])),
                "device_index": int(ast.literal_eval(call.args[1]) or 0),
            })
            continue
        if call.func.attr == "const":
            if not call.args:
                raise ValueError("reader.const requires a literal argument")
            try:
                value = ast.literal_eval(call.args[0])
            except (ValueError, TypeError):
                raise ValueError("reader.const requires a literal argument") from None
            inputs.append({
                "kind": "scalar",
                "name": input_name(call),
                "value": value,
            })
            continue
        if call.func.attr in {"opaque", "unsupported"}:
            inputs.append({
                "kind": "unsupported",
                "name": input_name(call),
                "reader_kind": call.func.attr,
            })
            continue

        if len(call.args) < 2:
            raise ValueError("reader.tensor requires storage and shape arguments")
        stride_node = call.args[2] if len(call.args) > 2 else keyword(call, "stride")
        dtype_node = keyword(call, "dtype")
        dtype_name = ast.unparse(dtype_node) if dtype_node is not None else "torch.float32"
        if not dtype_name.startswith("torch."):
            raise ValueError(f"unsupported reader dtype expression: {dtype_name}")
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported reader dtype: {dtype_name}")
        offset_node = keyword(call, "storage_offset")
        offset_expr = ast.unparse(offset_node) if offset_node is not None else "0"
        storage_key = ast.unparse(call.args[0])
        spec = {
            "kind": "tensor",
            "name": input_name(call),
            "shape": parse_dims(call.args[1]),
            "dtype": dtype,
            "stride": parse_dims(stride_node) if stride_node is not None else None,
            "device": storage_devices.get(storage_key),
            "storage_offset": parse_int(offset_expr),
            "storage_key": storage_key,
        }
        if offset_node is not None and any(
            isinstance(node, ast.Name) for node in ast.walk(offset_node)
        ):
            spec["symbolic_storage_offset"] = offset_expr
        inputs.append(spec)

    if inputs:
        inputs[0]["_symbol_bindings"] = bindings
    return _assign_alias_groups(inputs)


def _assign_alias_groups(input_specs):
    storage_counts = {}
    for spec in input_specs:
        storage_key = spec.get("storage_key")
        if storage_key:
            storage_counts[storage_key] = storage_counts.get(storage_key, 0) + 1
    alias_groups = {
        key: index
        for index, key in enumerate(
            sorted(key for key, count in storage_counts.items() if count > 1)
        )
    }
    for spec in input_specs:
        storage_key = spec.get("storage_key")
        if storage_key in alias_groups:
            spec["alias_group"] = alias_groups[storage_key]
    return input_specs


def _merge_input_specs(reader_specs, annotation_specs):
    if not reader_specs:
        return annotation_specs
    if not annotation_specs:
        return reader_specs
    reader_by_name = {
        spec.get("name"): spec for spec in reader_specs if spec.get("name")
    }
    if len(reader_by_name) != len(reader_specs):
        if len(reader_specs) != len(annotation_specs):
            raise ValueError(
                "reader inputs do not match the annotated forward signature"
            )
        pairs = list(zip(annotation_specs, reader_specs))
    else:
        pairs = list(
            (annotation, reader_by_name.get(annotation.get("name")))
            for annotation in annotation_specs
        )

    bindings = dict(reader_specs[0].get("_symbol_bindings", {}))
    for annotation, reader in pairs:
        if reader is None:
            continue
        expr = annotation.get("symbolic_expr")
        if (
            isinstance(expr, str)
            and re.fullmatch(r"[A-Za-z_]\w*", expr)
            and reader.get("value") is not None
        ):
            bindings[expr] = int(reader["value"])
        for item in annotation.get("symbolic_dims", []):
            symbol = item["symbol"]
            if re.fullmatch(r"[A-Za-z_]\w*", symbol):
                bindings[symbol] = int(reader["shape"][int(item["dim"])])

    def resolve_expr(expr: str) -> int:
        return _evaluate_int_expression(expr, bindings)

    merged = []
    for annotation, reader in pairs:
        if reader is None:
            raise ValueError(
                f"reader has no input for annotated parameter {annotation.get('name')!r}"
            )
        if reader.get("kind") == "unsupported":
            raise ValueError(
                f"reader cannot reconstruct {reader.get('reader_kind')} input "
                f"{annotation.get('name')!r}"
            )
        if annotation.get("kind") != reader.get("kind") and not (
            annotation.get("kind") == "scalar" and reader.get("kind") == "generator"
        ):
            raise ValueError(
                f"reader input kind does not match annotation for {annotation.get('name')!r}"
            )
        spec = dict(reader)
        spec["name"] = annotation.get("name")
        if spec.get("kind") == "tensor":
            if annotation.get("dtype") is not None and (
                str(spec.get("dtype")).removeprefix("torch.")
                != str(annotation["dtype"]).removeprefix("torch.")
            ):
                raise ValueError(
                    f"reader dtype does not match annotation for {spec['name']!r}"
                )
            symbolic_dims = {
                int(item["dim"]): item["symbol"]
                for item in annotation.get("symbolic_dims", [])
            }
            for index, actual in enumerate(spec["shape"]):
                expected = (
                    resolve_expr(symbolic_dims[index])
                    if index in symbolic_dims
                    else int(annotation["shape"][index])
                )
                if int(actual) != expected:
                    raise ValueError(
                        f"reader shape does not match annotation for {spec['name']!r}"
                    )
            if spec.get("stride") is None and annotation.get("stride") is not None:
                stride = list(annotation["stride"])
                for item in annotation.get("symbolic_stride_dims", []):
                    stride[int(item["dim"])] = resolve_expr(item["expr"])
                spec["stride"] = tuple(stride)
            if annotation.get("device"):
                spec["device"] = annotation["device"]
        for key in (
            "symbolic_expr",
            "symbolic_dims",
            "symbolic_stride_dims",
            "symbolic_storage_offset",
        ):
            if key in annotation:
                spec[key] = annotation[key]
        merged.append(spec)

    return _assign_alias_groups(merged)


def _dynamic_shapes_from_specs(input_specs):
    from torch.fx.experimental.dynamic_spec import (
        IntVar,
        ShapeVar,
        ShapesSpec,
        TensorSpec,
    )

    def expression_names(expr: str) -> set[str]:
        return {
            node.id
            for node in ast.walk(ast.parse(expr, mode="eval"))
            if isinstance(node, ast.Name)
            and node.id
            not in {
                "CeilToInt",
                "FloorDiv",
                "IntTrueDiv",
                "Max",
                "Min",
                "math",
                "max",
                "min",
            }
        }

    exprs = []
    shape_symbols = set()
    hints = {}
    for spec in input_specs:
        if spec.get("kind") == "symint" and spec.get("symbolic_expr"):
            exprs.append(spec["symbolic_expr"])
        if spec.get("symbolic_storage_offset"):
            exprs.append(spec["symbolic_storage_offset"])
        for item in spec.get("symbolic_dims", []):
            expr = item["symbol"]
            exprs.append(expr)
            names = expression_names(expr)
            shape_symbols.update(names)
            if len(names) == 1 and expr in names:
                hints.setdefault(expr, int(spec["shape"][int(item["dim"])]))

    symbol_names = {name for expr in exprs for name in expression_names(expr)}
    if not symbol_names:
        return None, False

    variables = {}
    for name in sorted(symbol_names):
        constructor = ShapeVar if name in shape_symbols else IntVar
        variables[name] = constructor(name, optimization_hint=hints.get(name))

    def convert(node):
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -convert(node.operand)
        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            right = convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left**right
        if isinstance(node, ast.Call):
            values = [convert(arg) for arg in node.args]
            if isinstance(node.func, ast.Name):
                if node.func.id == "FloorDiv" and len(values) == 2:
                    return values[0] // values[1]
                if node.func.id == "IntTrueDiv" and len(values) == 2:
                    return values[0] / values[1]
                if node.func.id == "CeilToInt" and len(values) == 1:
                    return math.ceil(values[0])
            if isinstance(node.func, ast.Name) and node.func.id in {"Max", "max"}:
                import torch

                return functools.reduce(torch.sym_max, values)
            if isinstance(node.func, ast.Name) and node.func.id in {"Min", "min"}:
                import torch

                return functools.reduce(torch.sym_min, values)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in {"ceil", "floor"}
            ):
                return getattr(math, node.func.attr)(*values)
        raise ValueError(f"unsupported symbolic shape expression: {ast.unparse(node)!r}")

    def parse_expr(expr: str):
        return convert(ast.parse(expr, mode="eval").body)

    params = {}
    for spec in input_specs:
        name = spec.get("name")
        if not name:
            raise ValueError("dynamic input is missing its parameter name")
        if spec.get("kind") == "tensor":
            symbolic_dims = {
                int(item["dim"]): item["symbol"]
                for item in spec.get("symbolic_dims", [])
            }
            if symbolic_dims:
                params[name] = TensorSpec([
                    parse_expr(symbolic_dims[index])
                    if index in symbolic_dims
                    else int(dim)
                    for index, dim in enumerate(spec["shape"])
                ])
        elif spec.get("kind") == "symint" and spec.get("symbolic_expr"):
            params[name] = parse_expr(spec["symbolic_expr"])
    return ShapesSpec(params), True


def _parse_input_shapes_from_annotations(content: str):
    """Parse input tensor shapes/dtypes from forward signature annotations.

    Annotations look like:
        primals_1: "f32[768, 768][768, 1]cuda:0"
        primals_2: "i64[1, 512][512, 1]cuda:0"
    """
    import re
    import torch

    dtype_map = {
        'f32': torch.float32,
        'f16': torch.float16,
        'bf16': torch.bfloat16,
        'i64': torch.int64,
        'i32': torch.int32,
        'i16': torch.int16,
        'i8': torch.int8,
        'u8': torch.uint8,
        'b8': torch.bool,
        'f64': torch.float64,
        'c64': torch.complex64,
        'c128': torch.complex128,
    }

    # Match the forward signature
    fwd_match = re.search(r'def forward\(self,?\s*(.*?)\):', content, re.DOTALL)
    if not fwd_match:
        return []

    def split_signature_args(sig: str) -> list[str]:
        args = []
        start = 0
        in_quote = False
        quote = ""
        for i, ch in enumerate(sig):
            if ch in {'"', "'"}:
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

    def parse_symbol_hint(annotation: str) -> int:
        match = re.search(r'Sym\((-?\d+)\)', annotation)
        return int(match.group(1)) if match else 32

    def parse_dim(dim: str) -> int:
        try:
            return int(dim)
        except ValueError:
            return 32

    sig = fwd_match.group(1)
    inputs = []
    for arg in split_signature_args(sig):
        match = re.match(r'\w+\s*:\s*"([^"]*)"', arg)
        if match is None:
            inputs.append({"kind": "scalar", "value": 1})
            continue

        annotation = match.group(1)
        if annotation.startswith("Sym("):
            inputs.append({"kind": "symint", "hint": parse_symbol_hint(annotation)})
            continue

        m = re.match(
            r'(\w+)\[([^\]]*)\](?:\[([^\]]*)\])?(.*)',
            annotation,
        )
        if m is None:
            inputs.append({"kind": "scalar", "value": 1})
            continue

        dtype_str, shape_str, stride_str, device_str = m.groups()
        dims = []
        for x in shape_str.split(','):
            x = x.strip()
            if not x:
                continue
            dims.append(parse_dim(x))
        stride = None
        if stride_str is not None:
            stride = []
            for x in stride_str.split(','):
                x = x.strip()
                if not x:
                    continue
                try:
                    stride.append(parse_dim(x))
                except ValueError:
                    stride = None
                    break
            if stride is not None:
                stride = tuple(stride)
        shape = tuple(dims)
        dtype = dtype_map.get(dtype_str, torch.float32)
        inputs.append({
            "shape": shape,
            "dtype": dtype,
            "stride": stride,
            "device": device_str.strip() or None,
        })

    return inputs


def _forward_takes_no_inputs(content: str) -> bool:
    import re

    fwd_match = re.search(r'def forward\(self,?\s*(.*?)\):', content, re.DOTALL)
    return fwd_match is not None and not fwd_match.group(1).strip()


def load_graph_module(
    graph_path: Path,
    *,
    tracing_mode: str = "real",
    annotation_path: Path | None = None,
):
    """Load a post-grad graph txt or fx_graph_runnable as an FX GraphModule.

    Handles two formats:
    - inductor_post_grad_graph_*.txt: has `class <lambda>(torch.nn.Module):`
    - fx_graph_runnable_*.txt: has `from torch.nn import *`, class Repro,
      __init__ with submodule constructors, and `if __name__` block.

    Strategy:
    1. Exec file content with globals that include torch.nn.* exports
    2. Strip __main__ blocks to avoid running the graph
    3. Replace `class <lambda>` with `class Repro`
    4. Trace the instantiated module with make_fx to produce a GraphModule

    ``tracing_mode="fake"`` traces without running the graph: inputs and
    tensor attributes are FakeTensors, so no device memory is allocated and
    value-dependent ops (repeat_interleave with output_size) cannot assert.
    Side-effect ops registered in mask_rules are stripped from the source
    when nothing reads their results.
    """
    import re
    import torch
    import torch.nn
    import torch.fx as fx
    import torch._inductor.inductor_prims  # noqa: F401 - registers prims.fma, inductor seeds

    content = graph_path.read_text()

    # Fix 1: Replace `class <lambda>` with `class Repro` (invalid Python syntax)
    content = content.replace("class <lambda>", "class Repro")

    # Fix 2: Strip `if __name__ == '__main__':` blocks to avoid running the graph.
    # These blocks compile/run the model which we don't want.
    content = re.sub(
        r'^if __name__\s*==\s*[\'"]__main__[\'"]:\s*\n(?:(?:[ \t]+.*)?\n)*',
        '', content, flags=re.MULTILINE
    )

    # Also strip `mod = Repro()` lines at module level that appear before __main__
    # (the file creates the instance itself which we'll do ourselves)
    content = re.sub(r'^mod\s*=\s*Repro\(\)\s*$', '', content, flags=re.MULTILINE)

    from mask_rules import strip_side_effect_source

    content, stripped, strip_blocked = strip_side_effect_source(content)

    try:
        from full_graph_harness import (
            _validate_tensor_layout,
            load_graph_class_from_source,
            parse_full_graph_inputs,
        )

        # Extract concrete values from the runnable and symbolic relationships from
        # the paired post-grad graph. The runnable alone has no shape annotations.
        reader_specs = _parse_input_shapes_from_reader(content)
        annotation_content = (
            annotation_path.read_text() if annotation_path is not None else content
        )
        if annotation_path is not None:
            annotation_content, annotation_stripped, annotation_blocked = (
                strip_side_effect_source(annotation_content)
            )
            if sorted(annotation_stripped) != sorted(stripped):
                raise ValueError(
                    "runnable and post-grad graph disagree on strippable "
                    f"side-effect ops: {stripped} vs {annotation_stripped}"
                )
            strip_blocked = strip_blocked + annotation_blocked
        side_effects = {
            "stripped": stripped,
            "blocked": [list(item) for item in strip_blocked],
        }
        annotation_specs = (
            parse_full_graph_inputs(annotation_content)
            if annotation_path is not None or not reader_specs
            else []
        )
        input_shapes = _merge_input_specs(reader_specs, annotation_specs)
        if not input_shapes:
            input_shapes = _parse_input_shapes_from_annotations(content)

        graph_cls = load_graph_class_from_source(content, graph_path)

        # Instantiate -- this runs __init__ which may register buffers/submodules
        instance = graph_cls()

        # Recreate saved self.* tensor attributes referenced by printed full graphs.
        # Their values do not affect partitioning; shape/dtype/device do.
        dtype_short_map = {
            'f32': torch.float32, 'f16': torch.float16,
            'bf16': torch.bfloat16, 'i64': torch.int64,
            'i32': torch.int32, 'i16': torch.int16, 'b8': torch.bool,
            'f64': torch.float64, 'i8': torch.int8, 'u8': torch.uint8,
        }

        def parse_shape(shape_str: str) -> tuple[int, ...]:
            dims = []
            for dim in shape_str.split(','):
                dim = dim.strip()
                if not dim:
                    continue
                try:
                    dims.append(int(dim))
                except ValueError:
                    dims.append(1)
            return tuple(dims)

        def collect_self_tensor_attrs(prefix: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
            info = {}
            pattern = re.compile(
                rf'\w+:\s*"(\w+)\[([^\]]*)\][^"]*"\s*=\s*self\.({prefix}\w*)'
            )
            for line in content.splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                dtype_str, shape_str, attr_name = match.groups()
                info[attr_name] = (
                    parse_shape(shape_str),
                    dtype_short_map.get(dtype_str, torch.float32),
                )
            return info

        tensor_attrs = {}
        # For inductor_post_grad_graph files: `_frozen_param*` buffers appear as:
        # `var: "dtype[shape][strides]device" = self._frozen_paramN`
        tensor_attrs.update(collect_self_tensor_attrs("_frozen_param"))
        # Saved full_graph_*.py files can also reference scalar/tensor constants:
        # `_tensor_constant0: "f32[]" = self._tensor_constant0`
        tensor_attrs.update(collect_self_tensor_attrs("_tensor_constant"))

        fake_mode = None
        if tracing_mode == "fake":
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.fx.experimental.symbolic_shapes import ShapeEnv

            fake_mode = FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(),
                static_shapes=True,
            )
        if fake_mode is not None:
            # Fake tensors need no GPU, so keep the graph's CUDA device.
            dev = torch.device('cuda:0')
        else:
            dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if tensor_attrs:
            for attr_name, (shape, dtype) in tensor_attrs.items():
                if hasattr(instance, attr_name):
                    continue
                if fake_mode is not None:
                    with fake_mode:
                        buf = torch.empty(shape, dtype=dtype, device=dev)
                else:
                    buf = torch.zeros(shape, dtype=dtype, device=dev)
                instance.register_buffer(attr_name, buf)

        # If already a GraphModule (has .graph), return directly
        if hasattr(instance, 'graph'):
            instance.meta["better_benchmark_side_effects"] = side_effects
            return instance

        # Otherwise, trace with make_fx to produce a proper GraphModule
        if not input_shapes and not _forward_takes_no_inputs(content):
            print(f"  No input shapes found for {graph_path.name}")
            return None

        from torch.fx.experimental.proxy_tensor import make_fx

        # Create inputs on CUDA (forward methods typically hardcode cuda:0 device)
        if fake_mode is None:
            instance = instance.to(dev)

        def normalize_input_spec(spec):
            if isinstance(spec, dict):
                return spec
            shape, dtype = spec
            return {"shape": shape, "dtype": dtype, "stride": None, "device": None}

        def make_tensor_from_spec(spec):
            if spec.get("kind") == "symint":
                value = spec.get("value", spec.get("hint"))
                if value is None:
                    raise ValueError(
                        f"reader has no concrete value for SymInt {spec.get('name')!r}"
                    )
                return int(value)
            if spec.get("kind") == "scalar":
                return spec.get("value", 1)
            if spec.get("kind") == "generator":
                if tracing_mode in {"symbolic", "fake"}:
                    return None
                device_type = spec.get("device_type", "cpu")
                device_index = int(spec.get("device_index", 0))
                if device_type == "cuda" and not torch.cuda.is_available():
                    device_type = "cpu"
                    device_index = 0
                return torch.Generator(device=torch.device(device_type, device_index))
            if spec.get("kind") == "unsupported":
                raise ValueError(
                    f"reader cannot reconstruct {spec.get('reader_kind')} input "
                    f"{spec.get('name')!r}"
                )

            shape = tuple(spec["shape"])
            dtype = spec["dtype"]
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype.removeprefix("torch."))
            stride = spec.get("stride")
            storage_offset = int(spec.get("storage_offset", 0) or 0)
            device_text = spec.get("device") or ""
            if device_text.startswith("cuda") and (
                fake_mode is not None or torch.cuda.is_available()
            ):
                tensor_dev = torch.device("cuda:0")
            elif device_text.startswith("cpu"):
                tensor_dev = torch.device("cpu")
            else:
                tensor_dev = dev

            if not dtype.is_floating_point and not dtype.is_complex:
                make_base = lambda size: torch.zeros(size, dtype=dtype, device=tensor_dev)
            elif str(dtype).startswith("torch.float8"):
                make_base = lambda size: torch.randn(
                    size, dtype=torch.float32, device=tensor_dev
                ).to(dtype)
            else:
                make_base = lambda size: torch.randn(size, dtype=dtype, device=tensor_dev)
            if fake_mode is not None:
                make_real = make_base

                def make_base(size):
                    with fake_mode:
                        return make_real(size)

            if stride is not None or storage_offset:
                if stride is None:
                    stride = []
                    running = 1
                    for dim in reversed(shape):
                        stride.append(running)
                        running *= max(int(dim), 1)
                    stride.reverse()
                if len(stride) != len(shape):
                    raise ValueError("input stride rank does not match shape rank")
                storage_size = storage_offset + sum(
                    s * (d - 1) for s, d in zip(stride, shape) if d > 1
                ) + 1
                return make_base((storage_size,)).as_strided(
                    shape,
                    stride,
                    storage_offset=storage_offset,
                )
            return make_base(shape)

        normalized_specs = [normalize_input_spec(spec) for spec in input_shapes]
        for spec in normalized_specs:
            if spec.get("kind", "tensor") != "tensor":
                continue
            _validate_tensor_layout(
                tuple(spec["shape"]),
                spec.get("stride"),
                int(spec.get("storage_offset", 0) or 0),
            )
        shared_storage = {}
        shared_storage_keys = {}
        for spec in normalized_specs:
            alias_group = spec.get("alias_group")
            if alias_group is None:
                continue
            shape = tuple(spec["shape"])
            stride = spec.get("stride")
            if stride is None:
                stride = []
                running = 1
                for dim in reversed(shape):
                    stride.append(running)
                    running *= max(int(dim), 1)
                stride.reverse()
            size = int(spec.get("storage_offset", 0) or 0) + sum(
                s * (d - 1) for s, d in zip(stride, shape) if d > 1
            ) + 1
            dtype = spec["dtype"]
            dtype_name = str(dtype).removeprefix("torch.")
            device_name = str(spec.get("device") or dev)
            if device_name.startswith("cuda"):
                device_name = "cuda"
            storage_key = (dtype_name, device_name)
            previous_key = shared_storage_keys.setdefault(alias_group, storage_key)
            if previous_key != storage_key:
                raise ValueError(
                    "aliased inputs with different dtypes or devices are unsupported"
                )
            current = shared_storage.get(alias_group)
            if current is None or current[0] < size:
                shared_storage[alias_group] = (size, spec)
        for alias_group, (size, spec) in list(shared_storage.items()):
            base_spec = dict(spec)
            base_spec.update({
                "alias_group": None,
                "shape": (size,),
                "storage_offset": 0,
                "stride": (1,),
            })
            shared_storage[alias_group] = make_tensor_from_spec(base_spec)

        inputs = []
        for spec in normalized_specs:
            alias_group = spec.get("alias_group")
            if alias_group is None:
                inputs.append(make_tensor_from_spec(spec))
                continue
            base = shared_storage[alias_group]
            shape = tuple(spec["shape"])
            stride = spec.get("stride")
            if stride is None:
                stride = []
                running = 1
                for dim in reversed(shape):
                    stride.append(running)
                    running *= max(int(dim), 1)
                stride.reverse()
            inputs.append(
                base.as_strided(
                    shape,
                    tuple(stride),
                    storage_offset=int(spec.get("storage_offset", 0) or 0),
                )
            )

        if tracing_mode == "symbolic":
            dynamic_shapes, captured_dynamic = _dynamic_shapes_from_specs(input_shapes)
            gm = make_fx(
                instance,
                tracing_mode="fake",
                dynamic_shapes=dynamic_shapes,
            )(*inputs)
        else:
            captured_dynamic = False
            gm = make_fx(instance, tracing_mode=tracing_mode)(*inputs)
        gm.meta["better_benchmark_input_specs"] = input_shapes
        gm.meta["better_benchmark_captured_dynamic"] = captured_dynamic
        gm.meta["better_benchmark_side_effects"] = side_effects
        if annotation_path is not None or any(spec.get("kind") == "generator" for spec in normalized_specs):
            gm.meta["better_benchmark_source_code"] = annotation_content.replace(
                "class <lambda>", "class Repro"
            )
        return gm

    except Exception as e:
        print(f"  Failed to load {graph_path.name}: {e}")
        return None


def process_graph_with_hook(gm, label: str, output_dir: Path, suite: str):
    """Run the capture hook's partitioning on a loaded GraphModule."""
    import torch
    from capture_hook import _CaptureState
    from merge_captures import merge_one_capture

    cap_dir = Path(tempfile.mkdtemp())
    state = _CaptureState(str(cap_dir), label=label)

    try:
        state.process_graph(gm)
        state.finalize()

        index_path = cap_dir / "index.json"
        if not index_path.exists():
            return 0

        n = merge_one_capture(cap_dir, output_dir, label, suite=suite, mode="infer")
        return n
    except Exception as e:
        print(f"  Process failed for {label}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=None,
                        help="GitHub Actions run ID to download from")
    parser.add_argument("--tlparse-dir", type=Path, default=None,
                        help="Already-downloaded tlparse directory")
    parser.add_argument("--suite", type=str, default="torchbench",
                        choices=["torchbench", "huggingface", "timm", "all"])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/tmp/scratch_space/better_benchmark/repros"))
    parser.add_argument("--list-only", action="store_true",
                        help="Just list available graphs, don't process")
    parser.add_argument("--use-runnables", action="store_true",
                        help="Use fx_graph_runnable files directly instead of post_grad_graph")
    args = parser.parse_args()

    if args.tlparse_dir:
        tlparse_dir = args.tlparse_dir
    elif args.run_id:
        tlparse_dir = Path(tempfile.mkdtemp()) / "tlparse"
        suite_filter = args.suite if args.suite != "all" else None
        download_tlparse(args.run_id, suite_filter, tlparse_dir)
    else:
        parser.error("Must provide --run-id or --tlparse-dir")

    if args.use_runnables:
        graphs = find_fx_graph_runnables(tlparse_dir)
        print(f"Found {len(graphs)} fx_graph_runnable files")
    else:
        graphs = find_post_grad_graphs(tlparse_dir)
        print(f"Found {len(graphs)} inductor_post_grad_graph files")

    if args.list_only:
        for g in graphs:
            print(f"  {g.relative_to(tlparse_dir)}")
        return

    import torch
    total_regions = 0
    for i, graph_path in enumerate(graphs):
        label = infer_model_name(graph_path) + f"_graph{i}"
        print(f"[{i+1}/{len(graphs)}] {graph_path.name} ({label})...", end=" ", flush=True)

        gm = load_graph_module(graph_path)
        if gm is None:
            print("SKIP (load failed)")
            continue

        import copy
        n = process_graph_with_hook(copy.deepcopy(gm), label, args.output_dir, args.suite)
        total_regions += n
        print(f"{n} regions")

    print(f"\nDone: {total_regions} regions from {len(graphs)} graphs")


if __name__ == "__main__":
    main()
