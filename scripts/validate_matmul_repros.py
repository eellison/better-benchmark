"""Reload every emitted MM repro/config and validate its FX category and identity."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from capture_hook import lift_shape_params, pattern_hash_for_subgraph
from matmul_partitioner import (
    CATEGORIES, classify_pattern_graph, is_matmul, is_reduction, mm_operands,
    singleton_reduction, tensor_shape,
)
from mm_replay import execution_hash
from repro_harness import load_shape_configs
from roundtrip_validation import fake_tensor_ctx


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "mm_repro")
    report = json.loads((output / "report.json").read_text())
    start = time.perf_counter()
    failures, points, files = [], 0, 0
    category_files = {c: 0 for c in CATEGORIES}
    for path in sorted((output / "canonical").glob("*/*/repro.py")):
        try:
            meta = json.loads(path.with_name("meta.json").read_text())
            category = meta["pattern_category"]
            assert category in CATEGORIES and path.parent.parent.name == category
            spec = importlib.util.spec_from_file_location("mm_validation_" + path.parent.name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            configs = load_shape_configs(str(path))
            declared = json.loads(path.with_name("shapes.json").read_text())["points"]
            assert len(configs) == len(declared) > 0
            with fake_tensor_ctx():
                for label, config in configs.items():
                    try:
                        model = module.Repro()
                        args = module.make_inputs(shape_config=config)
                        model(*args)
                        # Fresh inputs/model: mutation must not change the trace start state.
                        args = module.make_inputs(shape_config=config)
                        tensor_positions = [i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)]
                        model = module.Repro()
                        def bound(*tensors):
                            values = list(args)
                            for i, tensor in zip(tensor_positions, tensors):
                                values[i] = tensor
                            return model(*values)
                        gm = make_fx(bound, tracing_mode="fake")(*(args[i] for i in tensor_positions))
                        gm.graph.set_codegen(torch.fx.graph.CodeGen())
                        gm.recompile()
                        gm, _ = lift_shape_params(gm)
                        normal = category in {"prologue", "epilogue"}
                        assert classify_pattern_graph(gm, expected=category if normal else None) == category
                        if normal:
                            assert sum(is_matmul(n) for n in gm.graph.nodes) == 1
                            assert execution_hash(gm) == meta["pattern_hash"]
                        else:
                            assert pattern_hash_for_subgraph(gm, scalar_sensitive=True) == meta["pattern_hash"]
                        if category == "prologue":
                            # Bias-only work is not a matmul-input prologue.
                            mm = next(n for n in gm.graph.nodes if is_matmul(n))
                            pending, input_preparation = list(mm_operands(mm)), set()
                            while pending:
                                node = pending.pop()
                                if node in input_preparation:
                                    continue
                                input_preparation.add(node)
                                pending.extend(node.all_input_nodes)
                            assert any(n.op == "call_function" for n in input_preparation)
                        if category == "vertical":
                            shapes = {n: tensor_shape(n) for n in gm.graph.nodes}
                            assert not any(singleton_reduction(n, shapes)
                                           for n in gm.graph.nodes if is_reduction(n))
                        points += 1
                    except Exception:
                        failures.append({"repro": str(path), "shape": label, "error": traceback.format_exc()})
            files += 1
            category_files[category] += 1
        except Exception:
            failures.append({"repro": str(path), "error": traceback.format_exc()})
    if files != report["unique_repros"] or points != report["shape_points"]:
        failures.append({"error": "reloaded file/shape counts differ from report.json"})
    for category, count in category_files.items():
        if count != report["pattern_categories"][category]["repros"]:
            failures.append({"error": "category count differs", "category": category})
    print(json.dumps({"repros": files, "shape_points": points, "categories": category_files,
                      "failures": failures, "seconds": round(time.perf_counter() - start, 1)}, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
