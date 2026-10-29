"""Load the raw MM corpus and apply the final size limits."""

from __future__ import annotations

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
    """True if every measured size is within ``limits`` (a MatmulLimits)."""
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
    """Keep every record that fits; there is no count limit or sampling."""
    return [
        record
        for record in records
        if record["horizontal_shapes_match"] and admits(record["metrics"], limits)
    ]


def load_records(source):
    """Measure every raw repro so filter_patterns can apply the final limits.

    A record's metrics are the maximum over its shape points of both the
    partitioner's own numbers and a fresh FakeTensor trace of repro.py, so
    the limits hold for the code that is actually exported. Returns the
    records and the raw occurrence rows.
    """
    rows = json.loads((source / "occurrences.json").read_text())
    records = []
    torch.set_num_threads(1)
    paths = sorted((source / "canonical").glob("*/repro.py"))
    for index, path in enumerate(paths):
        pattern = json.loads(path.with_name("meta.json").read_text())["pattern_hash"]
        points = json.loads(path.with_name("shapes.json").read_text())["points"]
        spec = importlib.util.spec_from_file_location("filter_" + pattern, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        traced = Counter()
        horizontal_shapes_match = True
        with fake_tensor_ctx():
            for config in load_shape_configs(str(path)).values():
                gm = make_fx(module.Repro(), tracing_mode="fake")(
                    *make_inputs_from_config(config)
                )
                analysis = analyze_graph(gm)
                horizontal_shapes_match &= analysis["horizontal_shapes_match"]
                for key, value in analysis["metrics"].items():
                    traced[key] = max(traced[key], value)
        metrics = {
            key: max(traced[key], *(p["partition_metrics"][key] for p in points))
            for key in ["mm", "compute", "nodes", "inputs", "outputs", "contraction_depth"]
        }
        records.append(
            {
                "id": pattern,
                "directory": path.parent,
                "metrics": metrics,
                "horizontal_shapes_match": horizontal_shapes_match,
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Analyzed {index + 1}/{len(paths)} repros", flush=True)
    return records, rows
