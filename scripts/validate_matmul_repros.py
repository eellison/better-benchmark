"""Validate every MM repro and shape point by running it eagerly on a GPU.

Each repro runs in its own subprocess, so a CUDA device assert cannot poison
the rest of the sweep. For every shape point, repro_harness generates the
inputs on their recorded devices and Repro runs eagerly on those real tensors.
A second run with fresh inputs is traced by make_fx in real mode, which also
executes on the GPU (no FakeTensor), and the traced graph must keep the
repro's category, its single MM (prologue/epilogue) and its pattern hash.

Non-finite outputs from finite random inputs are reported but do not fail the
run: some patterns read statistics, such as a softmax's max and sum, as
independent random inputs.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/validate_matmul_repros.py [corpus] [--workers N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matmul_partitioner import CATEGORIES


def validate_repro(path: str) -> dict:
    """Run and retrace every shape point of one repro (in a worker process)."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils._pytree import tree_leaves

    from capture_hook import lift_shape_params, pattern_hash_for_subgraph
    from matmul_partitioner import classify_pattern_graph, is_matmul, mm_operands
    from repro_harness import load_shape_configs

    repro = Path(path)
    result = {"repro": path, "category": None, "points": 0, "failures": [], "nonfinite": []}
    try:
        meta = json.loads(repro.with_name("meta.json").read_text())
        category = result["category"] = meta["pattern_category"]
        assert category in CATEGORIES
        name = repro.parent.name
        assert name.startswith(category + "_") and name.endswith("_" + meta["pattern_hash"])
        spec = importlib.util.spec_from_file_location("mm_validation_" + name, repro)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configs = load_shape_configs(path)
        declared = json.loads(repro.with_name("shapes.json").read_text())["points"]
        assert len(configs) == len(declared) > 0
    except Exception:
        result["failures"].append({"error": traceback.format_exc()})
        return result
    normal = category in {"prologue", "epilogue"}
    for label, config in configs.items():
        try:
            with torch.no_grad():
                outputs = module.Repro()(*module.make_inputs(shape_config=config))
                torch.cuda.synchronize()
            nonfinite = [
                i for i, value in enumerate(tree_leaves(outputs))
                if isinstance(value, torch.Tensor)
                and (value.is_floating_point() or value.is_complex())
                and not bool(torch.isfinite(value).all())
            ]
            if nonfinite:
                result["nonfinite"].append({"shape": label, "outputs": nonfinite})
            del outputs
            # Fresh inputs and model: a mutation must not change the trace's start state.
            args = module.make_inputs(shape_config=config)
            positions = [i for i, arg in enumerate(args) if isinstance(arg, torch.Tensor)]
            model = module.Repro()

            def bound(*tensors):
                values = list(args)
                for i, tensor in zip(positions, tensors):
                    values[i] = tensor
                return model(*values)

            gm = make_fx(bound, tracing_mode="real")(*(args[i] for i in positions))
            gm.graph.set_codegen(torch.fx.graph.CodeGen())
            gm.recompile()
            gm, _ = lift_shape_params(gm)
            assert classify_pattern_graph(gm, expected=category if normal else None) == category
            if normal:
                assert sum(is_matmul(n) for n in gm.graph.nodes) == 1
            assert pattern_hash_for_subgraph(gm) == meta["pattern_hash"]
            if category == "prologue":
                # Bias-only work is not a matmul-input prologue.
                mm = next(n for n in gm.graph.nodes if is_matmul(n))
                pending, preparation = list(mm_operands(mm)), set()
                while pending:
                    node = pending.pop()
                    if node not in preparation:
                        preparation.add(node)
                        pending.extend(node.all_input_nodes)
                assert any(n.op == "call_function" for n in preparation)
            del gm, args, model
            result["points"] += 1
        except Exception:
            result["failures"].append({"shape": label, "error": traceback.format_exc()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", nargs="?", type=Path, default=ROOT / "mm_repro")
    parser.add_argument("--workers", type=int, default=8, help="repros validated in parallel")
    parser.add_argument("--json", type=Path, help="also write every repro's result here")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("validate_matmul_repros.py runs every repro eagerly and needs a CUDA GPU")
    report = json.loads((args.corpus / "report.json").read_text())
    paths = sorted(str(p) for p in (args.corpus / "canonical").glob("*/repro.py"))
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(validate_repro, path): path for path in paths}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception:  # the worker process died
                results.append({"repro": futures[future], "category": None, "points": 0,
                                "failures": [{"error": traceback.format_exc()}], "nonfinite": []})
    failures = [dict(f, repro=r["repro"]) for r in results for f in r["failures"]]
    nonfinite = [dict(n, repro=r["repro"]) for r in results for n in r["nonfinite"]]
    points = sum(r["points"] for r in results)
    categories = {c: sum(r["category"] == c for r in results) for c in CATEGORIES}
    if len(results) != report["unique_repros"] or points != report["shape_points"]:
        failures.append({"error": "validated repro/shape counts differ from report.json"})
    for category, count in categories.items():
        if count != report["pattern_categories"][category]["repros"]:
            failures.append({"error": "category count differs from report.json", "category": category})
    if args.json:
        args.json.write_text(json.dumps(sorted(results, key=lambda r: r["repro"]), indent=1))
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "repros": len(results),
        "shape_points": points,
        "categories": categories,
        "nonfinite_outputs": nonfinite,
        "failures": failures,
        "seconds": round(time.perf_counter() - start, 1),
    }, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
