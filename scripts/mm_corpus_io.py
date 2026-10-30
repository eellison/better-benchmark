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
from torch.fx.experimental.proxy_tensor import make_fx

from matmul_partitioner import is_computation, is_matmul, is_reduction, value_inputs
from repro_harness import load_shape_configs, make_inputs_from_config
from roundtrip_validation import fake_tensor_ctx


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def graph_metrics(gm):
    """Sizes of the actual standalone graph used by the corpus filters."""
    nodes = list(gm.graph.nodes)
    depth = {}
    for node in nodes:
        depth[node] = max((depth[a] for a in value_inputs(node)), default=0) + int(is_reduction(node))
    output = next(n for n in nodes if n.op == "output")
    return {
        "mm": sum(is_matmul(n) for n in nodes),
        "compute": sum(is_matmul(n) or is_computation(n) for n in nodes),
        "nodes": sum(n.op in {"call_function", "call_method", "get_attr"} for n in nodes),
        "inputs": sum(n.op == "placeholder" and isinstance(n.meta.get("val"), torch.Tensor) for n in nodes),
        "outputs": len(output.all_input_nodes),
        "contraction_depth": max(depth.values(), default=0),
    }


def admits(metrics, limits):
    """True if every measured size is within ``limits`` (matmul_partitioner.MatmulLimits)."""
    return all(
        metrics[key] <= cap
        for key, cap in (
            ("mm", limits.max_mm),
            ("compute", limits.max_compute),
            ("nodes", limits.max_nodes),
            ("inputs", limits.max_inputs),
            ("outputs", limits.max_outputs),
            ("contraction_depth", limits.max_stages),
        )
    )


def filter_patterns(records, limits):
    """Keep every record that fits; there is no count limit or sampling."""
    return [record for record in records if admits(record["metrics"], limits)]


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
        with fake_tensor_ctx():
            for config in load_shape_configs(str(path)).values():
                gm = make_fx(module.Repro(), tracing_mode="fake")(
                    *make_inputs_from_config(config)
                )
                for key, value in graph_metrics(gm).items():
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
            }
        )
        if (index + 1) % 100 == 0:
            print(f"Analyzed {index + 1}/{len(paths)} repros", flush=True)
    return records, rows
