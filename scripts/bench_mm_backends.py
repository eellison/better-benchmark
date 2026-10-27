#!/usr/bin/env python3
"""Compare GEMM backend policies using BB's existing parallel benchmark runner."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BACKENDS = ("ATEN", "TRITON", "NVGEMM", "CUTLASS")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def backend_payload(output, backend):
    paths = [output / f"{backend}.json", output / f"{backend}.pending.json"]
    payloads = [json.loads(path.read_text()) for path in paths if path.exists()]
    if not payloads:
        return {}
    if len(payloads) == 1:
        return payloads[0]
    results, failures, attempts = {}, {}, []
    elapsed = 0
    for payload in payloads:
        results.update({key: value for key, value in payload.items() if key.startswith("/")})
        failures.update(payload.get("__failures__", {}))
        elapsed += payload.get("__summary__", {}).get("elapsed_s", 0)
        metadata = dict(payload.get("_metadata", {}))
        attempts.extend(metadata.pop("attempt_metadata", []) or [metadata])
    failures = {key: value for key, value in failures.items() if key not in results}
    metadata = dict(payloads[-1].get("_metadata", {}), attempt_metadata=attempts,
                    n_repros=len(results), n_results=sum(len(row) for row in results.values()))
    total = json.loads((output / "run_config.json").read_text())["n_repros"]
    counts = {"total": total, "ok": len(results), "failed": len(failures), "skipped": 0}
    return {"_metadata": metadata, **results, "__failures__": failures,
            "__summary__": dict(counts, elapsed_s=elapsed, points=counts, repros=counts)}


def collect(output, repros, shape_labels):
    raw = {backend: backend_payload(output, backend) for backend in BACKENDS}
    rows = []
    wins = Counter()
    complete_wins = Counter()
    status_counts = {backend: Counter() for backend in BACKENDS}
    speedups = {backend: [] for backend in BACKENDS}
    per_repro = output / "per_repro"
    per_repro.mkdir(exist_ok=True)
    for repro in repros:
        row = {"repro": repro.parent.name, "shape": shape_labels[str(repro)], "backends": {}}
        for backend in BACKENDS:
            source = raw[backend]
            result = source.get(str(repro), {}).get("default")
            failure = source.get("__failures__", {}).get(str(repro))
            if result is not None:
                latency = result.get("compiled_us")
                if isinstance(latency, (int, float)) and math.isfinite(latency) and latency > 0:
                    record = {"status": "ok", "latency_us": latency, "bb_result": result}
                else:
                    record = {"status": "invalid_timing", "bb_result": result}
            elif failure is not None:
                record = {"status": "failed", "failure": failure}
            else:
                record = {"status": "pending"}
            row["backends"][backend] = record
            status_counts[backend][record["status"]] += 1
        valid = {backend: record["latency_us"] for backend, record in row["backends"].items()
                 if record["status"] == "ok"}
        row["successful_backends"] = len(valid)
        row["all_backends_succeeded"] = len(valid) == len(BACKENDS)
        row["winner_among_successful"] = min(valid, key=valid.get) if valid else None
        row["best_us"] = min(valid.values()) if valid else None
        if valid:
            wins[row["winner_among_successful"]] += 1
            if row["all_backends_succeeded"]:
                complete_wins[row["winner_among_successful"]] += 1
        aten = valid.get("ATEN")
        row["speedup_vs_aten"] = {backend: aten / latency for backend, latency in valid.items()} if aten else {}
        for backend, speedup in row["speedup_vs_aten"].items():
            speedups[backend].append(speedup)
        write_json(per_repro / f"{repro.parent.name}.json", row)
        rows.append(row)
    summary = {
        "n_repros": len(rows),
        "shape_policy": "first stored shape, one vote per repro",
        "latency_metric": "BB compiled_us: minimum of five CUDA graph timing rounds when capture succeeds",
        "numerics_checked": False,
        "backend_policy": "single max_autotune_gemm_backends value; autotune_fallback_to_aten=False",
        "scope": "whole-repro latency; backend setting controls GEMM autotune eligibility, not every non-GEMM kernel or unconditional lowering fallback",
        "all_backends_succeeded": sum(row["all_backends_succeeded"] for row in rows),
        "statuses": {backend: dict(status_counts[backend]) for backend in BACKENDS},
        "wins_among_successful": {backend: wins[backend] for backend in BACKENDS},
        "wins_all_four_successful": {backend: complete_wins[backend] for backend in BACKENDS},
        "paired_geomean_speedup_vs_aten": {
            backend: {"n_pairs": len(values),
                      "speedup": math.exp(sum(math.log(v) for v in values) / len(values)) if values else None}
            for backend, values in speedups.items()
        },
    }
    write_json(output / "comparison.json", {"summary": summary, "repros": rows})
    columns = ["repro", "shape", *[f"{b}_us" for b in BACKENDS],
               *[f"{b}_status" for b in BACKENDS], "winner_among_successful", "best_us",
               "successful_backends", "all_backends_succeeded"]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in columns if key in row}
            for backend, record in row["backends"].items():
                flat[f"{backend}_us"] = record.get("latency_us")
                flat[f"{backend}_status"] = record["status"]
            writer.writerow(flat)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "mm_repros/canonical")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mm_gemm_h100")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cutlass-dir", type=Path, default=Path("/home/jaeyeonw/pytorch-nullplay/third_party/cutlass"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repros = sorted(args.corpus.resolve().glob("*/repro.py"))
    if args.pilot:
        names = {"pointwise_24ce41abcd6c", "pointwise_3591805d23c9"}
        repros = [repro for repro in repros if repro.parent.name in names]
    if not repros or (not args.pilot and len(repros) != 387):
        parser.error(f"Expected {'two pilot' if args.pilot else '387'} repros, found {len(repros)}")
    from repro_harness import load_shape_configs
    import torch
    labels = {}
    fingerprint = hashlib.sha256()
    for repro in repros:
        configs = load_shape_configs(str(repro))
        if not configs:
            parser.error(f"No structured shapes: {repro}")
        labels[str(repro)] = next(iter(configs))
        fingerprint.update(repro.parent.name.encode())
        fingerprint.update(repro.read_bytes())
        fingerprint.update((repro.parent / "shapes.json").read_bytes())
    config = dict(torch._inductor.list_mode_options("max-autotune"))
    config.update(max_autotune_gemm=True, autotune_fallback_to_aten=False,
                  compile_threads=8, **{"cuda.cuda_cxx": "/usr/local/cuda/bin/nvcc"})
    metadata = {
        "corpus_sha256": fingerprint.hexdigest(), "torch_version": torch.__version__,
        "shape_policy": "first stored shape", "shape_labels": labels,
        "backends": list(BACKENDS), "inductor_config": config,
        "cutlass_dir": str(args.cutlass_dir.resolve()), "gpus": args.gpus,
        "n_repros": len(repros), "numerics_checked": False,
        "timing_runner": "scripts/bench_parallel.py", "workers_per_gpu": 1,
        "python": sys.executable,
        "input_seed": 0,
        "cuda_bin": "/usr/local/cuda/bin",
        "runner_sha256": hashlib.sha256((ROOT / "scripts/bench_parallel.py").read_bytes()).hexdigest(),
        "sweep_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "resumed_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "cache_root": str(args.cache_root.resolve()) if args.cache_root else str(output / "cache"),
    }
    inventory = output / "run_config.json"
    if inventory.exists() and json.loads(inventory.read_text()) != metadata:
        parser.error("Run configuration changed; use a fresh output directory")
    write_json(inventory, metadata)
    if args.resume_from:
        source = args.resume_from.resolve()
        previous = json.loads((source / "run_config.json").read_text())
        policy_keys = ("corpus_sha256", "torch_version", "shape_policy", "shape_labels",
                       "backends", "inductor_config", "cutlass_dir", "n_repros",
                       "numerics_checked", "input_seed", "runner_sha256")
        if any(previous.get(key) != metadata.get(key) for key in policy_keys):
            parser.error("Resume source has a different corpus or benchmark policy")
        for backend in BACKENDS:
            target = output / f"{backend}.json"
            if not target.exists():
                payload = backend_payload(source, backend)
                if payload:
                    write_json(target, payload)
    if args.collect_only:
        print(json.dumps(collect(output, repros, labels), indent=2))
        return
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for backend in BACKENDS:
        if state.get(backend, {}).get("complete"):
            continue
        previous = backend_payload(output, backend)
        completed = {key for key in previous if key.startswith("/")}
        completed.update(previous.get("__failures__", {}))
        remaining = [repro for repro in repros if str(repro) not in completed]
        pending_file = output / f"{backend}.pending.json"
        if previous:
            write_json(output / f"{backend}.json", previous)
        if pending_file.exists():
            pending_file.rename(output / f"{backend}.previous_{time.time_ns()}.json")
        if not remaining:
            state[backend] = {"complete": True, "statuses": collect(output, repros, labels)["statuses"][backend]}
            write_json(state_path, state)
            continue
        command = [sys.executable, "-u", str(ROOT / "scripts/bench_parallel.py"),
                   *map(str, remaining), "--device-kind", "H100", "--gpus", args.gpus,
                   "--workers-per-gpu", "1", "--strict-gpu-lock", "--no-cd",
                   "--input-seed", "0",
                   "--output", str(pending_file), "--tag", f"mm_{backend.lower()}"]
        options = dict(config, max_autotune_gemm_backends=backend)
        if backend == "CUTLASS":
            options["cutlass.cutlass_dir"] = str(args.cutlass_dir.resolve())
        for key, value in options.items():
            command.extend(["--inductor-config", f"{key}={value!r}"])
        cache_root = args.cache_root.resolve() if args.cache_root else output / "cache"
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(cache_root / backend),
                   PYTHONDONTWRITEBYTECODE="1",
                   PATH="/usr/local/cuda/bin:" + os.environ.get("PATH", ""))
        if backend == "CUTLASS":
            env["TORCHINDUCTOR_CUTLASS_DIR"] = str(args.cutlass_dir.resolve())
        state[backend] = {"complete": False, "started_unix": time.time(), "command": command,
                          "result_file": str(pending_file), "previous_tasks": len(completed)}
        write_json(state_path, state)
        print(f"Starting {backend}: {len(remaining)} pending repros on H100 GPUs {args.gpus}", flush=True)
        with (output / f"{backend}.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        payload = backend_payload(output, backend)
        if payload:
            write_json(output / f"{backend}.json", payload)
        if pending_file.exists():
            pending_file.rename(output / f"{backend}.attempt_{time.time_ns()}.json")
        summary = collect(output, repros, labels)
        statuses = summary["statuses"][backend]
        state[backend].update(returncode=result.returncode, finished_unix=time.time(),
                              complete=statuses.get("pending", 0) == 0, statuses=statuses)
        write_json(state_path, state)
        print(f"Finished {backend}: {statuses}", flush=True)
    print(json.dumps(collect(output, repros, labels), indent=2), flush=True)
    if any(not state.get(backend, {}).get("complete") for backend in BACKENDS):
        raise SystemExit("Incomplete backend run; inspect run_state.json and backend logs")


if __name__ == "__main__":
    main()
