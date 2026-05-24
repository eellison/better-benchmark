"""
Extract shapes from full_graph files and populate shapes.txt in canonical dirs.

For each full_graph file:
1. Load it as an FX GraphModule
2. Run _CaptureState.process_graph() to partition into regions
3. For each region, extract pattern_hash and _shapes_config
4. Write new shapes.txt entries to the corresponding canonical dir

This avoids the full merge pipeline — it ONLY adds shapes.txt entries.
"""
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from scripts.repartition_from_graphs import (
    find_all_full_graphs,
    infer_label_suite_mode,
    load_graph_module,
)
from capture_hook import _CaptureState


def find_canonical_dir_for_pattern(canonical_path: Path, pattern_hash: str) -> Path | None:
    """Find the canonical directory for a given pattern hash."""
    # Pattern hash is the last 12 chars of the directory name (after last underscore)
    for d in canonical_path.iterdir():
        if not d.is_dir():
            continue
        # Directory names are like: kind_patternhash (e.g., "amax_sum_1bad0f362efd")
        if d.name.endswith(f"_{pattern_hash}"):
            return d
        # Also check meta.json for exact match
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("pattern_hash") == pattern_hash:
                    return d
            except (json.JSONDecodeError, OSError):
                continue
    return None


def extract_shapes_config_from_file(filepath: Path) -> str | None:
    """Extract _shapes_config string from a generated repro file."""
    content = filepath.read_text()
    match = re.search(r'_shapes_config\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return None


def process_one_graph(graph_path: Path, canonical_path: Path,
                      pattern_dir_cache: dict) -> dict:
    """Process one full_graph file and return extracted shapes.

    Returns dict mapping pattern_hash -> (label, shapes_config)
    """
    label, suite, mode = infer_label_suite_mode(graph_path)

    gm = load_graph_module(graph_path)
    if gm is None:
        return {}

    results = {}

    with tempfile.TemporaryDirectory(prefix="shape_extract_") as tmp:
        state = _CaptureState(tmp, label=label, validate=False)
        try:
            state.process_graph(copy.deepcopy(gm))
            state.finalize()
        except Exception as e:
            print(f"  process_graph failed: {e}")
            return {}

        index_path = Path(tmp) / "index.json"
        if not index_path.exists():
            return {}

        entries = json.loads(index_path.read_text())
        for entry in entries:
            pattern_hash = entry.get("pattern_hash")
            shape_hash = entry.get("shape_hash")
            if not pattern_hash or not shape_hash:
                continue

            src_file = Path(entry["file"])
            if not src_file.exists():
                continue

            shapes_config = extract_shapes_config_from_file(src_file)
            if not shapes_config:
                continue

            config_key = f"{label.lower()}_{shape_hash[:8]}"
            results[f"{pattern_hash}:{config_key}"] = (pattern_hash, config_key, shapes_config)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-filter", type=str, default=None,
                        choices=["hf", "timm", "torchbench", "vllm", "genai"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N graphs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract but don't write")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "repros" / "models"
    canonical_path = PROJECT_ROOT / "repros" / "canonical"

    graph_files = find_all_full_graphs(models_dir)
    if args.suite_filter:
        graph_files = [f for f in graph_files if f"/{args.suite_filter}/" in str(f)]
    if args.limit:
        graph_files = graph_files[:args.limit]

    print(f"Processing {len(graph_files)} full_graph files...")
    print(f"Canonical dir: {canonical_path}")
    print(f"Existing shapes.txt count: {len(list(canonical_path.glob('*/shapes.txt')))}")
    print()

    # Build pattern hash -> canonical dir cache
    print("Building pattern hash -> directory cache...")
    pattern_dir_cache = {}
    for d in canonical_path.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                ph = meta.get("pattern_hash")
                if ph:
                    pattern_dir_cache[ph] = d
            except (json.JSONDecodeError, OSError):
                pass
    print(f"  Cached {len(pattern_dir_cache)} pattern directories")
    print()

    # Process all graphs
    all_shapes = {}  # pattern_hash -> {config_key: shapes_config}
    loaded = 0
    failed = 0

    for i, graph_path in enumerate(graph_files):
        rel = graph_path.relative_to(PROJECT_ROOT)
        print(f"[{i+1}/{len(graph_files)}] {rel}", end=" ", flush=True)

        try:
            results = process_one_graph(graph_path, canonical_path, pattern_dir_cache)
            if results:
                loaded += 1
                for key, (pattern_hash, config_key, shapes_config) in results.items():
                    if pattern_hash not in all_shapes:
                        all_shapes[pattern_hash] = {}
                    all_shapes[pattern_hash][config_key] = shapes_config
                print(f"-> {len(results)} regions")
            else:
                failed += 1
                print("SKIP")
        except Exception as e:
            failed += 1
            print(f"ERROR: {e}")

        # Free GPU memory periodically
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print(f"Extraction complete:")
    print(f"  Graphs processed:  {loaded}")
    print(f"  Graphs failed:     {failed}")
    print(f"  Unique patterns:   {len(all_shapes)}")
    print(f"  Total shape entries: {sum(len(v) for v in all_shapes.values())}")
    print()

    # Write shapes to shapes.txt files
    if args.dry_run:
        print("[DRY RUN] Not writing anything.")
        # Show a sample
        for ph, configs in list(all_shapes.items())[:5]:
            d = pattern_dir_cache.get(ph)
            print(f"  {ph} -> {d.name if d else 'NO DIR'}: {len(configs)} shapes")
        return

    new_entries_added = 0
    patterns_updated = 0
    patterns_without_dir = 0

    for pattern_hash, configs in all_shapes.items():
        canon_dir = pattern_dir_cache.get(pattern_hash)
        if canon_dir is None:
            patterns_without_dir += 1
            continue

        shapes_path = canon_dir / "shapes.txt"

        # Read existing entries
        existing_keys = set()
        if shapes_path.exists():
            for line in shapes_path.read_text().splitlines():
                if ":" in line:
                    key = line.split(":", 1)[0].strip()
                    existing_keys.add(key)

        # Add new entries
        new_lines = []
        for config_key, shapes_config in sorted(configs.items()):
            if config_key not in existing_keys:
                new_lines.append(f"{config_key}: {shapes_config}")
                new_entries_added += 1

        if new_lines:
            patterns_updated += 1
            with open(shapes_path, "a") as f:
                for line in new_lines:
                    f.write(line + "\n")

    print(f"Results:")
    print(f"  Patterns updated:       {patterns_updated}")
    print(f"  New entries added:       {new_entries_added}")
    print(f"  Patterns without dir:    {patterns_without_dir}")
    print(f"  Total shapes.txt files:  {len(list(canonical_path.glob('*/shapes.txt')))}")
    print()

    # Count multi-shape patterns
    multi_shape = 0
    for d in canonical_path.iterdir():
        sp = d / "shapes.txt"
        if sp.exists():
            lines = [l for l in sp.read_text().splitlines() if l.strip() and ":" in l]
            if len(lines) >= 2:
                multi_shape += 1
    print(f"  Patterns with multi-shape data: {multi_shape}")


if __name__ == "__main__":
    main()
