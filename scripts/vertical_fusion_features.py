"""Per-repro features that explain when a fused vertical kernel pays off.

    python scripts/vertical_fusion_features.py

For every repro with helion_vertical_result.json (new and old corpus), trace
the first shape point with FakeTensors and report:

  saved_ratio  bytes a one-kernel version avoids moving / bytes of the
               region's external inputs and outputs. A stage result that
               feeds a later stage must be written and re-read by separate
               kernels (only re-read if it is also a region output).
  stages       reduction stages (MMs and ordinary reductions)
together with fused/unfused from the result file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from mm_corpus_dedup import trace_repro  # noqa: E402
from matmul_partitioner import is_matmul, is_reduction, op_name  # noqa: E402
from repro_harness import load_shape_configs  # noqa: E402
from roundtrip_validation import fake_tensor_ctx  # noqa: E402


def nbytes(node):
    value = node.meta.get("val")
    values = value if isinstance(value, (tuple, list)) else [value]
    return sum(
        v.numel() * v.element_size() for v in values if isinstance(v, torch.Tensor)
    )


def features(directory: Path):
    path = directory / "repro.py"
    config = next(iter(load_shape_configs(str(path)).values()))
    with fake_tensor_ctx():
        gm, _infos, _positions = trace_repro(path, config)
    nodes = list(gm.graph.nodes)
    output = next(n for n in nodes if n.op == "output")
    outs = set(output.all_input_nodes)
    external = sum(nbytes(n) for n in nodes if n.op == "placeholder")
    external += sum(nbytes(n) for n in outs)
    stages = [n for n in nodes if is_reduction(n)]
    views = {
        "aten::view", "aten::reshape", "aten::_unsafe_view", "aten::permute",
        "aten::transpose", "aten::t", "aten::expand", "aten::clone",
        "aten::squeeze", "aten::unsqueeze", "prims::convert_element_type",
        "aten::_to_copy", "aten::alias",
    }
    # Stages each node depends on, and the separate-kernel boundaries: the
    # tensor a stage reads (views/casts stripped) when it comes from another
    # stage, except between consecutive ordinary reductions (one kernel).
    deps = {}
    for n in nodes:
        deps[n] = set().union(*(deps.get(a, set()) for a in n.all_input_nodes))
        if is_reduction(n):
            deps[n] = deps[n] | {n}
    outs_closure = set()
    for o in outs:
        x = o
        while op_name(x) in views and x.all_input_nodes:
            outs_closure.add(x)
            x = x.all_input_nodes[0]
        outs_closure.add(x)
    boundary = set()
    for t in stages:
        for a in t.all_input_nodes:
            x = a
            while op_name(x) in views and x.all_input_nodes:
                x = x.all_input_nodes[0]
            producers = deps.get(x, set())
            if not producers:
                continue
            if not is_matmul(t) and not any(is_matmul(s) for s in producers):
                continue
            boundary.add(x)
    saved = sum(nbytes(x) * (1 if x in outs_closure else 2) for x in boundary)
    return {
        "stages": len(stages),
        "external_MB": round(external / 1e6, 1),
        "saved_MB": round(saved / 1e6, 1),
        "saved_ratio": round(saved / external, 3) if external else None,
    }


def main():
    rows = []
    for pattern in (
        "benchmarks/vertical_helion/new/*/helion_vertical_result.json",
        "benchmarks/vertical_helion/old/*/helion_vertical_result.json",
    ):
        for result in sorted(ROOT.glob(pattern)):
            r = json.loads(result.read_text())
            f = features(result.parent)
            rows.append(
                (
                    "old" if "/old/" in pattern else "new",
                    r["repro"],
                    f,
                    r.get("fused_speedup_vs_unfused"),
                )
            )
    print("| corpus | repro | stages | external MB | saved MB | saved ratio | fused/unfused |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for corpus, repro, f, speedup in sorted(rows, key=lambda r: r[2]["saved_ratio"] or 0):
        print(
            f'| {corpus} | `{repro}` | {f["stages"]} | {f["external_MB"]} | {f["saved_MB"]} | '
            f'{f["saved_ratio"]} | {speedup} |'
        )


if __name__ == "__main__":
    main()
