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


def parse_make_inputs(repro_path: Path) -> list[dict]:
    """Parse make_inputs() from a repro file to extract input specs."""
    if not repro_path.exists():
        return []

    text = repro_path.read_text()

    # Find make_inputs function body
    match = re.search(r'def make_inputs\(\):\s*\n\s*return \[\n(.*?)\n\s*\]', text, re.DOTALL)
    if not match:
        return []

    body = match.group(1)
    specs = []

    for line in body.split("\n"):
        line = line.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue

        spec = {}

        # Detect dtype
        dtype_match = re.search(r'dtype=(torch\.\w+)', line)
        spec["dtype"] = dtype_match.group(1) if dtype_match else "torch.float32"

        # Detect device
        device_match = re.search(r"device='(\w+)'", line)
        spec["device"] = device_match.group(1) if device_match else "cuda"

        # Check for as_strided pattern
        strided_match = re.search(r'\.as_strided\(\[([^\]]+)\],\s*\[([^\]]+)\]\)', line)
        if strided_match:
            spec["shape"] = [int(x.strip()) for x in strided_match.group(1).split(",")]
            spec["stride"] = [int(x.strip()) for x in strided_match.group(2).split(",")]
        else:
            # Look for shape in various forms: [1,2,3] or (1,2,3)
            shape_match = re.search(r'(?:torch\.randint\(\d+,\s*\d+,\s*|torch\.randn\(|torch\.zeros\(|torch\.ones\()\[?([^\]\)]+)\]?', line)
            if shape_match:
                shape_str = shape_match.group(1).strip("[]() ")
                try:
                    spec["shape"] = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
                except ValueError:
                    continue
            else:
                continue
            spec["stride"] = None

        # Detect max_val for randint
        randint_match = re.search(r'torch\.randint\((\d+),\s*(\d+)', line)
        if randint_match:
            spec["max_val"] = int(randint_match.group(2))

        if spec.get("shape"):
            specs.append(spec)

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
    if stride:
        return f"T({shape}, {short_dtype}, stride={tuple(stride)})"
    return f"T({shape}, {short_dtype})"


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
