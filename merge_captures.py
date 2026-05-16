"""
Merge captured repros into the canonical set.

Takes a capture directory (produced by capture_hook.py or extract_reductions.py)
and upserts regions into the canonical repro set.

Usage:
    python merge_captures.py /tmp/captures/my_model --canonical-dir repros/ --model-name my_model_inference

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
)


def merge_one_capture(capture_dir: Path, canonical_dir: Path, model_name: str):
    """Merge a single capture directory into the canonical set."""
    index_path = capture_dir / "index.json"
    if not index_path.exists():
        print(f"  No index.json in {capture_dir}, skipping")
        return 0

    with open(index_path) as f:
        entries = json.load(f)

    canonical_path = canonical_dir / "canonical"
    manifest_path = canonical_dir / "manifest.json"
    canonical_path.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"version": 1, "models": {}}

    if model_name not in manifest["models"]:
        manifest["models"][model_name] = {"repros": []}

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

        # Update shapes.json
        shapes_path = repro_dir / "shapes.json"
        if shapes_path.exists():
            with open(shapes_path) as f:
                shapes_data = json.load(f)
        else:
            shapes_data = {"pattern_hash": pattern_hash, "configs": {}}

        config_key = f"{model_name.lower()}_{shape_hash[:8]}"
        if config_key not in shapes_data["configs"]:
            src_file = Path(entry["file"])
            input_specs = parse_make_inputs(src_file) if src_file.exists() else []
            shapes_data["configs"][config_key] = {
                "inputs": input_specs,
                "source_models": [model_name],
                "shape_hash": shape_hash,
            }
            with open(shapes_path, "w") as f:
                json.dump(shapes_data, f, indent=2)

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

        # Update manifest
        manifest["models"][model_name]["repros"].append({
            "pattern_hash": pattern_hash,
            "shape_config": config_key,
            "original_file": Path(entry["file"]).name,
        })
        merged += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge capture directories into canonical repro set")
    parser.add_argument("capture_dirs", nargs="+", type=Path,
                        help="One or more capture directories (containing index.json)")
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros"),
                        help="Path to canonical repro set root")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name for manifest. If not set, inferred from directory name.")
    args = parser.parse_args()

    total = 0
    for capture_dir in args.capture_dirs:
        if not capture_dir.is_dir():
            print(f"Skipping {capture_dir} (not a directory)")
            continue

        model_name = args.model_name or capture_dir.name
        print(f"Merging {capture_dir} as '{model_name}'...")
        n = merge_one_capture(capture_dir, args.canonical_dir, model_name)
        total += n
        print(f"  Merged {n} regions")

    print(f"\nTotal: {total} regions merged into {args.canonical_dir}/canonical/")


if __name__ == "__main__":
    main()
