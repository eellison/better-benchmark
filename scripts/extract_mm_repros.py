#!/usr/bin/env python3
"""Repartition saved BB full graphs with include_gemm=True; no oracle generation.

Without --execute, only inventory the source corpus. Extraction reuses BB's
_CaptureState and merge_one_capture, with one state per model (all its graphs).
Full graphs are replayed with their JSON input specs, not source annotations.
This is artifact recapture, not a rerun of the original benchmark models.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def load_definition(path):
    import torch
    import torch._inductor.inductor_prims  # Registers prims used by saved graphs.
    from full_graph_harness import FullGraphDefinition, load_full_graph_sidecar

    sidecar = load_full_graph_sidecar(path)
    if not {"inputs", "tensor_attrs"}.issubset(sidecar):
        raise RuntimeError(f"Missing structured input metadata: {path}")
    namespace = {name: getattr(torch.nn, name) for name in dir(torch.nn)
                 if not name.startswith("_")}
    namespace.update(torch=torch, fx=torch.fx, inf=math.inf, nan=math.nan,
                     device=torch.device, tensor=torch.tensor)
    loaded = runpy.run_path(str(path), init_globals=namespace)
    classes = [value for value in loaded.values()
               if isinstance(value, type) and issubclass(value, torch.nn.Module)
               and value.__module__ == loaded["__name__"]]
    if len(classes) != 1:
        raise RuntimeError(f"Expected one saved graph class: {path}")
    parameters = list(inspect.signature(classes[0].forward).parameters)[1:]
    specs = sidecar["inputs"]
    if (len(parameters) != len(specs)
            or any(spec.get("name") not in {None, name}
                   for name, spec in zip(parameters, specs))):
        raise RuntimeError(f"Sidecar/forward input mismatch: {path}")
    # Compact scalar/SymInt entries are positional and omit their names.
    for name, spec in zip(parameters, specs):
        spec["name"] = name
    return FullGraphDefinition(
        path=path, graph_cls=classes[0], input_specs=sidecar["inputs"],
        tensor_attrs=sidecar["tensor_attrs"], forward_takes_no_inputs=not parameters,
        metadata={"source": sidecar.get("source", {}), "sidecar": sidecar},
    )


def capture_model(model_dir, work_dir, validate=True):
    import torch
    from capture_hook import _CaptureState, graph_node_accounting
    from roundtrip_validation import trace_full_graph_fake

    state = _CaptureState(str(work_dir / "raw"), label=model_dir.name,
                          validate=validate, include_gemm=True)
    saved = work_dir / "graphs"
    saved.mkdir(parents=True, exist_ok=True)
    for graph in sorted(model_dir.glob("full_graph_*.py")):
        print(f"Capturing {graph}", flush=True)
        definition = load_definition(graph)
        gm = trace_full_graph_fake(definition)
        try:
            state.process_graph(gm)
        finally:
            state.finalize()
        if state.dropped:
            raise RuntimeError(f"BB dropped regions: {state.dropped}")
        # Preserve original graph/input metadata. Account for the new policy
        # separately; the old round-trip verdict belongs to the source capture.
        shutil.copy2(graph, saved / graph.name)
        metadata = json.loads(graph.with_suffix(".meta.json").read_text())
        original_accounting = metadata.pop("node_accounting", None)
        original_roundtrip = metadata.pop("roundtrip", None)
        metadata["source_capture"] = {
            "path": str(graph), "node_accounting": original_accounting,
            "roundtrip": original_roundtrip,
        }
        metadata["partition_policy"] = {"include_gemm": True}
        metadata["node_accounting"] = graph_node_accounting(gm, include_gemm=True)
        write_json(saved / graph.with_suffix(".meta.json").name, metadata)
        del gm, definition
        torch.cuda.empty_cache()


def verify_repro(repro_dir):
    import torch
    from repro_harness import load_shape_configs

    loaded = runpy.run_path(str(repro_dir / "repro.py"), init_globals={
        "device": torch.device, "inf": math.inf, "nan": math.nan,
    })
    configs = load_shape_configs(str(repro_dir / "repro.py"))
    if not configs:
        raise RuntimeError(f"No shape configs: {repro_dir}")
    for name, config in configs.items():
        print(f"Eager validating {repro_dir.name}/{name}", flush=True)
        inputs = loaded["make_inputs"](shape_config=config)
        with torch.no_grad():
            output = loaded["Repro"]()(*inputs)
        torch.cuda.synchronize()
        del inputs, output


def summarize_corpus(output):
    """Count ops in real FX retraces of installed repros, using FakeTensors only."""
    import torch
    from collections import Counter
    from statistics import median
    import torch._inductor.inductor_prims  # Registers saved Inductor prims.
    from torch.fx.experimental.proxy_tensor import make_fx
    from capture_hook import GEMM_OP_NAMES
    from repro_harness import load_shape_configs
    from roundtrip_validation import fake_tensor_ctx

    rows, failures = [], []
    for repro in sorted((output / "canonical").glob("*/repro.py")):
        try:
            loaded = runpy.run_path(str(repro), init_globals={
                "device": torch.device, "inf": math.inf, "nan": math.nan,
            })
            configs = load_shape_configs(str(repro))
            if not configs:
                raise RuntimeError("No structured shape configs")
            with fake_tensor_ctx(), torch.no_grad():
                inputs = loaded["make_inputs"](shape_config=next(iter(configs.values())))
                gm = make_fx(loaded["Repro"](), tracing_mode="fake")(*inputs)
            names = [node.target._schema.name for node in gm.graph.nodes
                     if node.op == "call_function"
                     and isinstance(node.target, torch._ops.OpOverload)]
            rows.append({"repro": repro.parent.name,
                         "aten_ops": sum(name.startswith("aten::") for name in names),
                         "gemm_ops": sum(name in GEMM_OP_NAMES for name in names),
                         "call_function_ops": sum(n.op == "call_function" for n in gm.graph.nodes),
                         "gemm_types": sorted(set(name for name in names if name in GEMM_OP_NAMES)),
                         "gemm_op_counts": dict(sorted(Counter(name for name in names if name in GEMM_OP_NAMES).items()))})
            print(f"Counted {repro.parent.name}", flush=True)
            del inputs, gm
        except Exception:
            failures.append({"repro": str(repro), "error": traceback.format_exc()})
    gemm_rows = [row for row in rows if row["gemm_ops"]]
    summary = {
        "n_repro_files": len(rows) + len(failures), "n_counted": len(rows),
        "n_gemm_repros": len(gemm_rows),
        "mean_aten_ops": sum(r["aten_ops"] for r in rows) / len(rows) if rows else None,
        "mean_gemm_ops": sum(r["gemm_ops"] for r in rows) / len(rows) if rows else None,
        "gemm_repros_mean_aten_ops": sum(r["aten_ops"] for r in gemm_rows) / len(gemm_rows) if gemm_rows else None,
        "gemm_repros_mean_gemm_ops": sum(r["gemm_ops"] for r in gemm_rows) / len(gemm_rows) if gemm_rows else None,
        "max_aten_ops": max((r["aten_ops"] for r in rows), default=0),
        "min_aten_ops": min((r["aten_ops"] for r in rows), default=0),
        "median_aten_ops": median(r["aten_ops"] for r in rows) if rows else None,
        "max_gemm_ops": max((r["gemm_ops"] for r in rows), default=0),
        "counting": "One vote per unique repro.py; first shape config; fake FX retrace; no eager numerics",
        "gemm_op_names": sorted(GEMM_OP_NAMES), "graphs": rows, "failures": failures,
    }
    gemm_totals = Counter()
    for row in rows:
        gemm_totals.update(row["gemm_op_counts"])
    summary["gemm_op_totals"] = dict(sorted(gemm_totals.items()))
    previous_path = output / "graph_stats.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if "filtering" in previous:
            summary["filtering"] = previous["filtering"]
    write_json(output / "graph_stats.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"graphs", "failures"}}, indent=2))
    return 1 if failures else 0


def run_worker(args, kind, target, log, work=None):
    command = [sys.executable, str(Path(__file__).resolve()), kind, str(target)]
    if work is not None:
        command.extend(["--work-dir", str(work)])
    if args.skip_validation:
        command.append("--skip-validation")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               PYTHONDONTWRITEBYTECODE="1")
    with log.open("w") as stream:
        try:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            return f"Timeout after {args.timeout}s; see {log}"
    return None if result.returncode == 0 else f"Exit {result.returncode}; see {log}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "repros/models")
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repros")
    parser.add_argument("--model", help="One relative model directory for a pilot run")
    parser.add_argument("--gpu", default="0", help="One physical GPU for eager validation")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per worker")
    parser.add_argument("--execute", action="store_true", help="Write the new corpus")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Extract only; defer capture-time and installed eager validation")
    parser.add_argument("--capture-model", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--verify-repro", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stats-only", action="store_true", help="Count installed repro FX ops with FakeTensors")
    args = parser.parse_args()
    if args.capture_model:
        capture_model(args.capture_model, args.work_dir, validate=not args.skip_validation)
        return 0
    if args.verify_repro:
        verify_repro(args.verify_repro)
        return 0
    if args.stats_only:
        return summarize_corpus(args.output.resolve())

    from recapture_full_graphs import infer_target
    source, output = args.source.resolve(), args.output.resolve()
    if (output == ROOT / "repros" or output == source
            or output in source.parents or source in output.parents):
        parser.error("Output must be a separate corpus outside the source tree")
    if not args.gpu.isdecimal() or args.timeout <= 0:
        parser.error("Use one numeric GPU index and a positive timeout")
    graphs = sorted(source.rglob("full_graph_*.py"))
    groups = {}
    for graph in graphs:
        relative = graph.parent.relative_to(source).as_posix()
        if args.model is None or relative == args.model:
            groups.setdefault(relative, []).append(graph)
    if not groups:
        parser.error("No matching full graphs")
    missing = [str(g) for gs in groups.values() for g in gs
               if not g.with_suffix(".meta.json").exists()]
    if missing:
        parser.error(f"Missing JSON sidecars: {missing}")
    print(f"{len(groups)} models, {sum(map(len, groups.values()))} full graphs")
    print(f"Source: {source}\nOutput: {output}\ninclude_gemm=True; validate={not args.skip_validation}; no oracles")
    if not args.execute:
        for key, model_graphs in groups.items():
            print(f"  {key}: {len(model_graphs)} graphs")
        print("Inventory only. Add --execute after reviewing this script.")
        return 0

    from gpu_lock import gpu_lock
    from merge_captures import merge_one_capture
    import torch

    policy = {"source": str(source), "include_gemm": True,
              "torch_version": torch.__version__,
              "capture_hook_sha256": hashlib.sha256((ROOT / "capture_hook.py").read_bytes()).hexdigest()}
    policy_path = output / "extraction_policy.json"
    if output.exists() and any(output.iterdir()):
        if not policy_path.exists() or json.loads(policy_path.read_text()) != policy:
            parser.error("Existing output has a different or unknown extraction policy")
    write_json(policy_path, policy)
    report_path = output / "extraction_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"models": {}, "verification": {}}
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    lock = nullcontext() if args.skip_validation else gpu_lock(args.gpu, label="BB include_gemm corpus extraction")
    with lock:
        for i, (key, model_graphs) in enumerate(groups.items(), 1):
            digest = hashlib.sha256()
            for graph in model_graphs:
                digest.update(graph.name.encode())
                digest.update(graph.read_bytes())
                digest.update(graph.with_suffix(".meta.json").read_bytes())
            fingerprint = digest.hexdigest()
            previous = report["models"].get(key, {})
            if previous.get("status") == "ok" and previous.get("source_sha256") == fingerprint:
                continue
            if previous.get("status") == "ok":
                parser.error(f"Source changed for {key}; use a fresh output directory")
            print(f"[{i}/{len(groups)}] {key}", flush=True)
            work = output / ".work" / key
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            log = logs / (key.replace("/", "__") + ".log")
            error = run_worker(args, "--capture-model", model_graphs[0].parent, log, work)
            merged = 0
            if error is None:
                target = infer_target(model_graphs[0], source)
                model_output = output / "models" / key
                model_output.mkdir(parents=True, exist_ok=True)
                try:
                    for saved in (work / "graphs").iterdir():
                        shutil.copy2(saved, model_output / saved.name)
                    merged = merge_one_capture(work / "raw", output, target.model,
                                               suite=target.suite, mode=target.mode)
                except Exception:
                    error = traceback.format_exc()
            report["models"][key] = {
                "status": "failed" if error else "ok", "error": error,
                "source_sha256": fingerprint, "graphs": len(model_graphs),
                "merged_pattern_shape_pairs": merged,
                "eager_validation": "deferred" if args.skip_validation else "capture_passed",
            }
            write_json(report_path, report)
            if error is None:
                shutil.rmtree(work)
            else:
                print(error, flush=True)

        # Optional eager validation consumes installed artifacts after merge.
        canonical = output / "canonical"
        for repro in ([] if args.skip_validation else sorted(canonical.glob("*/repro.py"))):
            log = logs / ("verify__" + repro.parent.name + ".log")
            error = run_worker(args, "--verify-repro", repro.parent, log)
            report["verification"][repro.parent.name] = {"status": "failed" if error else "ok", "error": error}
            write_json(report_path, report)
            print(f"Verify {repro.parent.name}: {'failed' if error else 'ok'}", flush=True)
    failures = [entry for name in ("models", "verification") for entry in report.get(name, {}).values()
                if entry["status"] != "ok"]
    print(f"Finished; {len(failures)} failures. Report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
