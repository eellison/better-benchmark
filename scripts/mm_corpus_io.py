"""Shared size and input-loading helpers for the core MM corpus generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from mm_corpus_features import analyze_graph
from torch.fx.experimental.proxy_tensor import make_fx

from repro_harness import load_shape_configs, make_inputs_from_config
from roundtrip_validation import fake_tensor_ctx


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def admits(metrics, limits):
    return all(
        metrics[key] <= cap
        for key, cap in (
            ("mm", limits.max_mm),
            ("compute", limits.max_compute),
            ("nodes", limits.max_nodes),
            ("inputs", limits.max_inputs),
            ("outputs", limits.max_outputs),
            ("contraction_depth", limits.max_reductions_chain),
        )
    )


def filter_patterns(records, limits):
    """Keep all passing regions; there is no corpus-count limit or sampling."""
    return [
        record
        for record in records
        if record["horizontal_shapes_match"] and admits(record["metrics"], limits)
    ]


def load_records(source, cache_path):
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    classifier = (
        Path(__file__).with_name("mm_corpus_features.py").read_bytes()
        + (ROOT / "matmul_partitioner.py").read_bytes()
    )
    occurrence_rows = json.loads((source / "occurrences.json").read_text())
    occurrences = Counter(
        r["pattern_hash"] for row in occurrence_rows for r in row["regions"]
    )
    records = []
    torch.set_num_threads(1)
    paths = sorted((source / "canonical").glob("*/repro.py"))
    for index, path in enumerate(paths):
        shape_bytes = path.with_name("shapes.json").read_bytes()
        digest = hashlib.sha256(
            classifier + path.read_bytes() + shape_bytes
        ).hexdigest()
        meta = json.loads(path.with_name("meta.json").read_text())
        pattern = meta["pattern_hash"]
        points = json.loads(shape_bytes)["points"]
        stored = cache.get(pattern)
        if stored is None or stored["digest"] != digest:
            spec = importlib.util.spec_from_file_location("filter_" + pattern, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            coverage, actual = set(), Counter()
            horizontal_shapes_match = True
            with fake_tensor_ctx():
                for config in load_shape_configs(str(path)).values():
                    gm = make_fx(module.Repro(), tracing_mode="fake")(
                        *make_inputs_from_config(config)
                    )
                    analysis = analyze_graph(gm)
                    coverage.update(analysis["coverage"])
                    horizontal_shapes_match &= analysis["horizontal_shapes_match"]
                    for key, value in analysis["metrics"].items():
                        actual[key] = max(actual[key], value)
            stored = {
                "digest": digest,
                "coverage": sorted(coverage),
                "actual_metrics": dict(actual),
                "horizontal_shapes_match": horizontal_shapes_match,
            }
            cache[pattern] = stored
        metrics = {
            key: max(
                stored["actual_metrics"][key],
                *(p["partition_metrics"][key] for p in points),
            )
            for key in [
                "mm",
                "compute",
                "nodes",
                "inputs",
                "outputs",
                "contraction_depth",
            ]
        }
        records.append(
            {
                "id": pattern,
                "directory": path.parent,
                "coverage": set(stored["coverage"]),
                "metrics": metrics,
                "horizontal_shapes_match": stored["horizontal_shapes_match"],
                "shape_points": len(points),
                "occurrences": occurrences[pattern],
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Analyzed {index + 1}/{len(paths)} repros", flush=True)
    write_json(cache_path, cache)
    return records, occurrence_rows
