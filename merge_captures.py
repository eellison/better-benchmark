"""
Merge captured repros into the canonical set.

Takes a capture directory (produced by capture_hook.py or extract_reductions.py)
and upserts regions into the canonical repro set.

Usage:
    python merge_captures.py /tmp/captures/my_model --canonical-dir repros/ --model-name my_model --suite hf --mode train

    # Merge multiple capture directories
    python merge_captures.py /tmp/captures/model_a /tmp/captures/model_b --canonical-dir repros/
"""
import argparse
import json
from pathlib import Path

from canonicalize_repros import (
    extract_docstring,
    extract_imports,
    extract_make_inputs_body,
    extract_repro_class,
    generate_canonical_repro,
    parse_make_inputs,
    _spec_to_T,
)

def _format_compact_config(label: str, input_specs: list[dict]) -> str:
    """Format input specs as compact one-liner: label: (T([...], f32), S([...]), ...)"""
    parts = []
    for spec in input_specs:
        if spec.get("kind") == "shape":
            parts.append(f"S({spec['dims']})")
        else:
            parts.append(_spec_to_T(spec))
    return f"{label}: ({', '.join(parts)})"


def _write_model_json(canonical_dir: Path, model_name: str, patterns: list[str],
                      suite: str = "other", mode: str | None = None):
    """Write per-model JSON file with pattern list."""
    models_dir = canonical_dir / "models"
    if mode:
        out_dir = models_dir / suite / mode
    else:
        out_dir = models_dir / suite
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{model_name}.json"

    # Merge with existing if present
    if out_file.exists():
        existing = json.loads(out_file.read_text())
        patterns = sorted(set(existing["patterns"] + patterns))
    else:
        patterns = sorted(set(patterns))

    out_file.write_text(json.dumps({"patterns": patterns}, indent=2) + "\n")
    return out_file


def _infer_suite_mode(model_name: str) -> tuple[str, str | None, str]:
    """Infer suite, mode, and clean name from a model label."""
    name = model_name

    if name.startswith("dynamo_"):
        name = name.replace("dynamo_", "")
        if "_inference" in name:
            return "hf", "infer", name.replace("_inference", "")
        return "hf", "train", name

    if name.startswith("hf_"):
        name = name.replace("hf_", "")
        if "_train" in name:
            return "hf", "train", name.replace("_train", "")
        return "hf", "infer", name

    if name.startswith("timm_"):
        name = name.replace("timm_", "")
        if "_training" in name:
            return "timm", "train", name.replace("_training", "")
        if "_inference" in name:
            return "timm", "infer", name.replace("_inference", "")
        return "timm", "infer", name

    if name.startswith("vllm_"):
        name = name.replace("vllm_", "")
        if "_inference" in name:
            name = name.replace("_inference", "")
        return "vllm", None, name

    if name.startswith("genai_") or name.startswith("tritonbench_"):
        return "genai", None, name

    if name.startswith("tlparse_"):
        return "tlparse", None, name.replace("tlparse_", "")

    # torchvision or other
    if "_training" in name:
        return "torchvision", "train", name.replace("_training", "")
    if "_inference" in name:
        return "torchvision", "infer", name.replace("_inference", "")

    return "other", None, name


def merge_one_capture(capture_dir: Path, canonical_dir: Path, model_name: str,
                      suite: str | None = None, mode: str | None = None):
    """Merge a single capture directory into the canonical set."""
    index_path = capture_dir / "index.json"
    if not index_path.exists():
        print(f"  No index.json in {capture_dir}, skipping")
        return 0

    with open(index_path) as f:
        entries = json.load(f)

    canonical_path = canonical_dir / "canonical"
    canonical_path.mkdir(parents=True, exist_ok=True)

    # Infer suite/mode if not provided
    if suite is None:
        suite, inferred_mode, clean_name = _infer_suite_mode(model_name)
        if mode is None:
            mode = inferred_mode
    else:
        clean_name = model_name

    merged_patterns = []
    merged = 0

    for entry in entries:
        pattern_hash = entry.get("pattern_hash")
        if not pattern_hash:
            continue

        shape_hash = entry.get("shape_hash", "unknown")
        kind = entry.get("kind", "region")
        reduction_types = entry.get("reduction_types", [])
        kind_label = "_".join(reduction_types[:3]) if reduction_types else kind
        dir_name = f"{kind_label}_{pattern_hash}"
        repro_dir = canonical_path / dir_name
        repro_dir.mkdir(parents=True, exist_ok=True)

        # Update shapes.txt (compact T()/S() format)
        shapes_path = repro_dir / "shapes.txt"
        config_key = f"{model_name.lower()}_{shape_hash[:8]}"

        # Check if this config already exists
        existing_lines = shapes_path.read_text().splitlines() if shapes_path.exists() else []
        if not any(line.startswith(f"{config_key}:") for line in existing_lines):
            src_file = Path(entry["file"])
            input_specs = parse_make_inputs(src_file) if src_file.exists() else []
            if input_specs:
                compact_line = _format_compact_config(config_key, input_specs)
                with open(shapes_path, "a") as f:
                    f.write(compact_line + "\n")

        # Update meta.json
        meta_path = repro_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if model_name not in meta.get("models", []):
                meta["models"].append(model_name)
                meta["models"].sort()
                meta["n_models"] = len(meta["models"])
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        else:
            meta = {
                "pattern_hash": pattern_hash,
                "kind": kind,
                "reduction_types": reduction_types,
                "n_ops": entry.get("n_ops"),
                "origin_ops": entry.get("origin_ops", []),
                "n_models": 1,
                "models": [model_name],
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Write canonical repro.py if it doesn't exist
        repro_py = repro_dir / "repro.py"
        if not repro_py.exists():
            src_file = Path(entry["file"])
            if src_file.exists():
                try:
                    repro_class = extract_repro_class(src_file)
                    docstring = extract_docstring(src_file)
                    imports = extract_imports(src_file)
                    fallback = extract_make_inputs_body(src_file)
                    if repro_class:
                        code = generate_canonical_repro(repro_class, docstring, imports, fallback)
                        repro_py.write_text(code)
                except Exception as e:
                    print(f"  Warning: could not generate canonical repro for {dir_name}: {e}")

        merged_patterns.append(pattern_hash)
        merged += 1

    # Write per-model JSON
    if merged_patterns:
        out_file = _write_model_json(canonical_dir, clean_name, merged_patterns, suite, mode)
        print(f"  Model JSON: {out_file}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge capture directories into canonical repro set")
    parser.add_argument("capture_dirs", nargs="+", type=Path,
                        help="One or more capture directories (containing index.json)")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros"),
                        help="Path to canonical repro set root")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for manifest. If not set, inferred from directory name.")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["hf", "timm", "vllm", "torchvision", "genai", "tritonbench", "torchbench", "other"],
                        help="Suite to categorize this model under.")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["train", "infer"],
                        help="Mode (training or inference). If not set, inferred from model name.")
    args = parser.parse_args()

    total = 0
    for capture_dir in args.capture_dirs:
        if not capture_dir.is_dir():
            print(f"Skipping {capture_dir} (not a directory)")
            continue

        model_name = args.model_name or capture_dir.name
        print(f"Merging {capture_dir} as '{model_name}'...")
        n = merge_one_capture(capture_dir, args.canonical_dir, model_name,
                              suite=args.suite, mode=args.mode)
        total += n
        print(f"  Merged {n} regions")

    print(f"\nTotal: {total} regions merged into {args.canonical_dir}/canonical/")


if __name__ == "__main__":
    main()
