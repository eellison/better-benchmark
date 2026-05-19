"""
Build canonical repro set + manifest from existing per-model aten_repros.

Scans output/aten_repros/*/index.json, groups by pattern_hash, and writes:
  repros/canonical/<pattern_name>_<pattern_hash>/
    repro.py       - rewritten to use repro_harness
    shapes.json    - all shape configs across models
    meta.json      - pattern metadata
  repros/manifest.json - model → repro mapping

Usage:
    python canonicalize_repros.py [--aten-dir output/aten_repros] [--out-dir repros]
"""
import argparse
import ast
import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path


def scan_all_indexes(aten_dir: Path) -> list[dict]:
    entries = []
    for model_dir in sorted(aten_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        index_file = model_dir / "index.json"
        if not index_file.exists():
            continue
        model_name = model_dir.name
        with open(index_file) as f:
            index = json.load(f)
        for entry in index:
            entry["_model"] = model_name
            entries.append(entry)
    return entries


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _call_name(node) -> str | None:
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _kw(call: ast.Call, name: str):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _dtype_from_call(call: ast.Call, default: str = "torch.float32") -> str:
    dtype_node = _kw(call, "dtype")
    dtype_name = _call_name(dtype_node)
    if dtype_name and dtype_name.startswith("torch."):
        return dtype_name
    return default


def _shape_from_arg(node):
    value = _literal(node)
    if isinstance(value, int):
        return [value]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and all(isinstance(dim, int) for dim in value):
        return value
    return None


def _unwrap_view_call(node):
    """Return (base_call, shape, stride) for torch factory calls with optional view ops."""
    shape = None
    stride = None
    cur = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        attr = cur.func.attr
        if attr == "reshape" and cur.args:
            shape = _shape_from_arg(cur.args[0])
            cur = cur.func.value
            continue
        if attr == "as_strided" and len(cur.args) >= 2:
            shape = _shape_from_arg(cur.args[0])
            stride_value = _literal(cur.args[1])
            if isinstance(stride_value, tuple):
                stride_value = list(stride_value)
            if isinstance(stride_value, list):
                stride = stride_value
            cur = cur.func.value
            continue
        break
    return cur if isinstance(cur, ast.Call) else None, shape, stride


def _parse_tensor_factory(node) -> dict | None:
    call, view_shape, stride = _unwrap_view_call(node)
    if call is None:
        return None

    name = _call_name(call.func)
    if name not in {"torch.randn", "torch.randint", "torch.randperm"}:
        return None

    if name == "torch.randperm":
        if not call.args:
            return None
        max_val = _literal(call.args[0])
        if not isinstance(max_val, int):
            return None
        shape = view_shape or [max_val]
        spec = {
            "shape": shape,
            "dtype": _dtype_from_call(call, default="torch.int64"),
            "device": "cuda",
            "stride": stride,
            "gen": {"kind": "permutation", "size": max_val},
        }
        return spec

    if name == "torch.randint":
        if len(call.args) < 3:
            return None
        min_val = _literal(call.args[0])
        max_val = _literal(call.args[1])
        shape = view_shape or _shape_from_arg(call.args[2])
        if not isinstance(min_val, int) or not isinstance(max_val, int) or shape is None:
            return None
        spec = {
            "shape": shape,
            "dtype": _dtype_from_call(call, default="torch.int64"),
            "device": "cuda",
            "stride": stride,
            "gen": {"kind": "index", "low": min_val, "high": max_val},
        }
        return spec

    shape_node = call.args[0] if call.args else _kw(call, "size")
    shape = view_shape or (_shape_from_arg(shape_node) if shape_node is not None else None)
    if shape is None:
        return None
    return {
        "shape": shape,
        "dtype": _dtype_from_call(call),
        "device": "cuda",
        "stride": stride,
    }


def _parse_make_inputs_source(repro_path: Path) -> list[dict]:
    try:
        tree = ast.parse(repro_path.read_text())
    except SyntaxError:
        return []

    funcs = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_default_make_inputs", "make_inputs"}
    }
    func = funcs.get("_default_make_inputs") or funcs.get("make_inputs")
    if func is None:
        return []

    return_node = next((node for node in ast.walk(func) if isinstance(node, ast.Return)), None)
    if return_node is None or not isinstance(return_node.value, (ast.List, ast.Tuple)):
        return []

    specs = []
    for elt in return_node.value.elts:
        if isinstance(elt, ast.Call):
            spec = _parse_tensor_factory(elt)
            if spec is None:
                return []
            specs.append(spec)
            continue

        value = _literal(elt)
        if isinstance(value, list):
            specs.append({"kind": "shape", "dims": value})
        elif isinstance(value, int):
            specs.append({"kind": "shape", "dims": [value]})
        else:
            return []
    return specs


def parse_make_inputs(repro_path: Path) -> list[dict]:
    """Extract input specs from a captured repro."""
    if not repro_path.exists():
        return []

    source_specs = _parse_make_inputs_source(repro_path)
    if source_specs:
        return source_specs

    import importlib.util
    import math
    import torch

    try:
        spec = importlib.util.spec_from_file_location("repro_parse", str(repro_path))
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        if hasattr(mod, '_default_make_inputs'):
            inputs = mod._default_make_inputs()
        elif hasattr(mod, 'make_inputs'):
            inputs = mod.make_inputs()
        else:
            return []
    except Exception:
        return []

    specs = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            spec = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": "cuda",
            }
            if not item.is_contiguous():
                spec["stride"] = list(item.stride())
            else:
                spec["stride"] = None
            specs.append(spec)
        elif isinstance(item, list):
            specs.append({"kind": "shape", "dims": item})
        elif isinstance(item, int):
            specs.append({"kind": "shape", "dims": [item]})

    return specs


def extract_repro_class(repro_path: Path) -> str:
    """Extract the class Repro(...) and forward() from a repro file."""
    text = repro_path.read_text()

    # Find class definition through end of forward method
    class_match = re.search(
        r'(# The extracted FX graph subgraph:\n)?class Repro\(.*?\n(    def forward\(self.*?\n(?:        .*\n)*?)(?=\n\ndef |\nclass |\Z)',
        text,
        re.MULTILINE
    )
    if class_match:
        return class_match.group(0).rstrip()

    # Fallback: grab from "class Repro" to the next top-level def/class
    lines = text.split("\n")
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("class Repro("):
            start = i
        elif start is not None and (line.startswith("def ") or (line.startswith("class ") and i > start)):
            end = i
            break
    if start is not None:
        end = end or len(lines)
        return "\n".join(lines[start:end]).rstrip()

    return ""


def extract_docstring(repro_path: Path) -> str:
    """Extract the module docstring from a repro file."""
    text = repro_path.read_text()
    match = re.match(r'^(""".*?""")', text, re.DOTALL)
    if match:
        return match.group(1)
    return '"""Canonical repro."""'


def extract_imports(repro_path: Path) -> list[str]:
    """Extract import lines (before class Repro)."""
    text = repro_path.read_text()
    lines = text.split("\n")
    imports = []
    for line in lines:
        if line.startswith("class Repro("):
            break
        if line.startswith("import ") or line.startswith("from "):
            # Skip inductor_config import since harness handles it
            if "inductor_config" in line:
                continue
            imports.append(line)
    return imports


def build_shape_config_name(model_name: str) -> str:
    name = model_name.lower()
    name = name.replace("dynamo_", "")
    name = name.replace("vllm_", "vllm_")
    return name


def generate_canonical_repro(
    repro_class_code: str,
    docstring: str,
    imports: list[str],
    default_make_inputs_code: str,
) -> str:
    """Generate a canonical repro.py that uses repro_harness."""

    import_block = "\n".join(imports)

    return f'''{docstring}
import sys
from pathlib import Path

{import_block}

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

{repro_class_code}


def _default_make_inputs():
{default_make_inputs_code}


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
'''


def extract_make_inputs_body(repro_path: Path) -> str:
    """Get the raw make_inputs return statement for the fallback.

    Returns the body indented at 4 spaces (suitable for a function def).
    """
    text = repro_path.read_text()
    lines = text.split("\n")
    in_func = False
    body_lines = []
    for line in lines:
        if line.startswith("def make_inputs():"):
            in_func = True
            continue
        if in_func:
            if line and not line[0].isspace() and line.strip():
                break
            body_lines.append(line)

    # Strip trailing empty lines
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    if not body_lines:
        return "    return []"

    # Dedent to base level, then re-indent to 4 spaces
    min_indent = float("inf")
    for line in body_lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    if min_indent == float("inf"):
        min_indent = 0

    result = []
    for line in body_lines:
        if line.strip():
            result.append("    " + line[min_indent:])
        else:
            result.append("")
    return "\n".join(result)


def _generation_expr(spec: dict) -> str | None:
    gen = spec.get("gen")
    if gen is None:
        if spec.get("constraint") == "permutation":
            gen = {"kind": "permutation", "size": spec.get("max_val")}
        elif spec.get("max_val") is not None:
            gen = {"kind": "index", "low": 0, "high": spec["max_val"]}
    if gen is None:
        return None

    if gen.get("kind") == "permutation":
        size = gen.get("size")
        return "Perm()" if size is None else f"Perm({size})"

    if gen.get("kind") == "index":
        low = int(gen.get("low", 0))
        high = int(gen["high"])
        if low == 0:
            return f"Index({high})"
        return f"Index({high}, low={low})"

    return None


def _spec_to_T(spec: dict) -> str:
    """Convert an input spec dict to compact T() notation."""
    shape = spec["shape"]
    dtype_str = spec["dtype"].replace("torch.", "")
    # Shorten common dtypes
    dtype_map = {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        "float64": "f64", "int64": "i64", "int32": "i32",
        "int16": "i16", "int8": "i8", "bool": "b8", "uint8": "u8",
    }
    short_dtype = dtype_map.get(dtype_str, dtype_str)
    stride = spec.get("stride")
    kwargs = []
    if stride:
        kwargs.append(f"stride={tuple(stride)}")
    gen = _generation_expr(spec)
    if gen is not None:
        kwargs.append(f"gen={gen}")
    suffix = f", {', '.join(kwargs)}" if kwargs else ""
    return f"T({shape}, {short_dtype}{suffix})"


def _inputs_to_line(input_specs: list[dict]) -> str:
    """Convert a list of input specs to one compact line."""
    parts = [_spec_to_T(s) for s in input_specs]
    return f"(({', '.join(parts)}), {{}})"


def _write_shapes_txt(path: Path, all_input_tuples: list[list[dict]]):
    """Write shapes.txt with deduped input tuples and counts."""
    from collections import Counter
    # Deduplicate by converting to canonical string representation
    line_counts = Counter()
    for specs in all_input_tuples:
        if specs:
            line = _inputs_to_line(specs)
            line_counts[line] += 1

    lines = []
    for line, cnt in line_counts.most_common():
        lines.append(f"cnt: {cnt}, {line}")

    path.write_text("\n".join(lines) + "\n" if lines else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aten-dir", type=Path, default=Path("output/aten_repros"))
    parser.add_argument("--out-dir", type=Path, default=Path("repros"))
    args = parser.parse_args()

    aten_dir = args.aten_dir
    out_dir = args.out_dir
    canonical_dir = out_dir / "canonical"
    manifest_path = out_dir / "manifest.json"

    entries = scan_all_indexes(aten_dir)
    print(f"Scanned {len(entries)} repro entries across models")

    by_pattern = defaultdict(list)
    for entry in entries:
        ph = entry.get("pattern_hash")
        if ph:
            by_pattern[ph].append(entry)

    print(f"Found {len(by_pattern)} unique pattern hashes")
    canonical_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"version": 1, "models": {}}
    pattern_summary = []

    for pattern_hash, group in sorted(by_pattern.items(), key=lambda x: -len(x[1])):
        first = group[0]
        reduction_types = first.get("reduction_types", [])
        kind_label = "_".join(reduction_types[:3]) if reduction_types else first.get("kind", "region")
        dir_name = f"{kind_label}_{pattern_hash}"
        canonical_path = canonical_dir / dir_name
        canonical_path.mkdir(parents=True, exist_ok=True)

        # Collect all input specs and build manifest entries
        all_input_tuples = []
        models_using = set()

        for entry in group:
            model = entry["_model"]
            models_using.add(model)
            shape_hash = entry.get("shape_hash", "unknown")

            repro_path = Path(entry["file"])
            input_specs = parse_make_inputs(repro_path)

            if input_specs:
                all_input_tuples.append(input_specs)

            if model not in manifest["models"]:
                manifest["models"][model] = {"repros": []}
            manifest["models"][model]["repros"].append({
                "pattern_hash": pattern_hash,
                "shape_hash": shape_hash,
                "region_index": entry.get("node_name", ""),
                "original_file": Path(entry["file"]).name,
            })

        # Write shapes.txt in compact T() format (deduped with counts)
        _write_shapes_txt(canonical_path / "shapes.txt", all_input_tuples)

        # Write meta.json
        meta = {
            "pattern_hash": pattern_hash,
            "kind": first.get("kind"),
            "reduction_types": first.get("reduction_types", []),
            "n_ops": first.get("n_ops"),
            "origin_ops": first.get("origin_ops", []),
            "n_models": len(models_using),
            "models": sorted(models_using),
        }
        with open(canonical_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Generate canonical repro.py from first instance
        src_file = Path(first["file"])
        repro_py = canonical_path / "repro.py"
        if src_file.exists():
            repro_class = extract_repro_class(src_file)
            docstring = extract_docstring(src_file)
            imports = extract_imports(src_file)
            fallback_body = extract_make_inputs_body(src_file)

            if repro_class:
                code = generate_canonical_repro(repro_class, docstring, imports, fallback_body)
                repro_py.write_text(code)

        pattern_summary.append({
            "pattern_hash": pattern_hash,
            "dir": dir_name,
            "n_models": len(models_using),
            "n_shapes": len(all_input_tuples),
            "kind": kind_label,
        })

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCanonical repros written to {canonical_dir}/")
    print(f"Manifest written to {manifest_path}")
    print(f"\nTop patterns by model coverage:")
    for p in sorted(pattern_summary, key=lambda x: -x["n_models"])[:15]:
        print(f"  {p['dir']:50s}  {p['n_models']:2d} models, {p['n_shapes']:2d} shapes")


if __name__ == "__main__":
    main()
