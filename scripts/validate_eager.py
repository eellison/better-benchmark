"""Validate canonical repros in eager mode.

Runs each repro/shape in an isolated subprocess so CUDA device asserts do not
poison the whole validation sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _find_repros(paths: list[Path]) -> list[Path]:
    repros: list[Path] = []
    for path in paths:
        if path.is_file():
            repros.append(path)
        else:
            repros.extend(path.glob("*/repro.py"))
    return sorted(set(repros))


def _load_benchmark_set(path: Path, canonical_dir: Path) -> list[Path]:
    data = json.loads(path.read_text())
    repros = []
    for entry in data.get("patterns", []):
        name = entry.get("name")
        if name:
            repro = canonical_dir / name / "repro.py"
            if repro.exists():
                repros.append(repro)
    for entry in data.get("benchmarks", []):
        name = entry.get("repro")
        if name:
            repro = canonical_dir / name / "repro.py"
            if repro.exists():
                repros.append(repro)
    return sorted(set(repros))


def _shape_labels(repro: Path, all_shapes: bool) -> list[str | None]:
    if not all_shapes:
        return [None]

    sys.path.insert(0, str(ROOT))
    from repro_harness import load_shape_configs

    configs = load_shape_configs(str(repro))
    return list(configs) if configs else [None]


def _load_repro_module(repro: Path):
    spec = importlib.util.spec_from_file_location(f"eager_repro_{abs(hash(repro))}", repro)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {repro}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_inputs(module, repro: Path, shape_label: str | None):
    from repro_harness import load_shape_configs, make_inputs_from_config, make_inputs_safely

    if shape_label is None:
        return make_inputs_safely(module.make_inputs)

    configs = load_shape_configs(str(repro))
    inputs = make_inputs_from_config(configs[shape_label])
    default_inputs = make_inputs_safely(module.make_inputs)
    if len(inputs) >= len(default_inputs):
        return inputs

    merged = []
    config_idx = 0
    import torch

    for default in default_inputs:
        if isinstance(default, (list, int)) and not isinstance(default, torch.Tensor):
            merged.append(default)
        elif config_idx < len(inputs):
            merged.append(inputs[config_idx])
            config_idx += 1
        else:
            merged.append(default)
    return merged


def _run_child(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    import torch

    repro = Path(args.child_repro)
    shape_label = None if args.child_shape == "__default__" else args.child_shape
    result = {
        "repro": str(repro),
        "shape": shape_label or "default",
        "ok": False,
        "error": None,
        "elapsed_s": None,
    }

    start = time.time()
    try:
        module = _load_repro_module(repro)
        inputs = _make_inputs(module, repro, shape_label)
        model = module.Repro()
        with torch.no_grad():
            out = model(*inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        del out, inputs, model
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=8)
    finally:
        result["elapsed_s"] = time.time() - start

    Path(args.child_output).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


def _run_one(
    repro: Path,
    shape_label: str | None,
    gpu: str,
    timeout_s: int,
) -> dict:
    with tempfile.NamedTemporaryFile(prefix="eager_validate_", suffix=".json", delete=False) as tmp:
        output = Path(tmp.name)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        __file__,
        "--child-repro",
        str(repro),
        "--child-shape",
        shape_label or "__default__",
        "--child-output",
        str(output),
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        if output.exists() and output.stat().st_size:
            result = json.loads(output.read_text())
        else:
            result = {
                "repro": str(repro),
                "shape": shape_label or "default",
                "ok": False,
                "error": f"child exited {proc.returncode} without result",
            }
        result["returncode"] = proc.returncode
        if proc.stdout:
            result["output_tail"] = proc.stdout[-4000:]
        return result
    except subprocess.TimeoutExpired as exc:
        return {
            "repro": str(repro),
            "shape": shape_label or "default",
            "ok": False,
            "error": f"timeout after {timeout_s}s",
            "output_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "elapsed_s": time.time() - start,
        }
    finally:
        try:
            output.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repros in eager mode")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--benchmark-set", type=Path)
    parser.add_argument("--canonical-dir", type=Path, default=Path("repros/canonical"))
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-repro")
    parser.add_argument("--child-shape")
    parser.add_argument("--child-output")
    args = parser.parse_args()

    if args.child_repro:
        return _run_child(args)

    if args.benchmark_set:
        repros = _load_benchmark_set(args.benchmark_set, args.canonical_dir)
    else:
        repros = _find_repros(args.paths or [args.canonical_dir])

    tasks = [
        (repro, shape)
        for repro in repros
        for shape in _shape_labels(repro, args.all_shapes)
    ]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus cannot be empty")

    print(
        f"Eager-validating {len(tasks)} repro points "
        f"({len(repros)} repros) across {args.max_workers} workers on GPUs {','.join(gpus)}"
    )

    results = []
    ok = 0
    failed = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _run_one,
                repro,
                shape,
                gpus[i % len(gpus)],
                args.timeout_s,
            ): (repro, shape)
            for i, (repro, shape) in enumerate(tasks)
        }
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            name = Path(result["repro"]).parent.name
            shape = result["shape"]
            elapsed = result.get("elapsed_s") or 0.0
            if result["ok"]:
                ok += 1
                print(f"  [{idx}/{len(tasks)}] OK   {elapsed:5.1f}s  {name} [{shape}]")
            else:
                failed += 1
                print(
                    f"  [{idx}/{len(tasks)}] FAIL {elapsed:5.1f}s  "
                    f"{name} [{shape}]: {result.get('error')}"
                )

    summary = {
        "ok": ok,
        "failed": failed,
        "elapsed_s": time.time() - start,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"[output] Wrote {args.output}")

    print(f"Done: {ok} ok, {failed} failed in {summary['elapsed_s']:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
