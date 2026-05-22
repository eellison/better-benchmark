"""Smoke-test bench_parallel worker recovery with intentionally bad repros."""

from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_parallel import _locked_worker, _persistent_worker_script, _results_payload


def _write_repro(path: Path, source: str) -> Path:
    repro_dir = path.parent
    repro_dir.mkdir(parents=True)
    path.write_text(source)
    return path


VALID_REPRO = """
import torch


class Repro(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + 1.0


def make_inputs():
    return (torch.randn((16,), device="cuda"),)
"""


NORMAL_FAILURE_REPRO = """
import torch


class Repro(torch.nn.Module):
    def forward(self):
        raise RuntimeError("intentional non-cuda repro failure")


def make_inputs():
    return ()
"""


CUDA_FAILURE_REPRO = """
import torch


class Repro(torch.nn.Module):
    def forward(self):
        raise RuntimeError("CUDA intentional repro failure")


def make_inputs():
    return ()
"""


def _fake_persistent_worker_script(gpu_idx: str, args_dict: dict) -> str:
    return r'''
import fcntl
import json
import os
import sys


def _lock_path():
    lock_dir = os.environ.get("INDUCTOR_GPU_BENCH_LOCK_DIR")
    if not lock_dir:
        return None
    os.makedirs(lock_dir, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return os.path.join(lock_dir, f"gpu_{gpu}.lock")


for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break
    if line.startswith("PREFETCH:"):
        continue

    name = os.path.basename(os.path.dirname(line))
    if name.startswith("ok_"):
        print(json.dumps({
            "default": {
                "compiled_us": 1.0,
                "coord_descent_us": None,
                "memcopy_sol_us": 1.0,
                "total_bytes": 4,
                "gap_default": 1.0,
                "gap_cd": None,
            }
        }), flush=True)
    elif name == "normal_fail":
        print("ERROR: intentional non-cuda repro failure", flush=True)
    elif name == "cuda_fail":
        path = _lock_path()
        if path is not None:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
        print("CUDA_ERROR: intentional device failure", flush=True)
        sys.exit(1)
    else:
        print(f"ERROR: unknown fake repro {name}", flush=True)
'''


def _strict_lock_stress_worker(
    script_prefix: str,
    lock_dir: str,
    iterations: int,
    barrier,
    results,
) -> None:
    os.environ["INDUCTOR_GPU_BENCH_LOCK"] = "1"
    os.environ["INDUCTOR_GPU_BENCH_LOCK_DIR"] = lock_dir
    namespace = {"__name__": "bench_parallel_generated_worker_stress"}
    try:
        exec(script_prefix, namespace)
        from torch._inductor.runtime import benchmarking as inductor_benchmarking

        for _ in range(iterations):
            with namespace["gpu_setup_lock"]():
                barrier.wait(timeout=60)
                with inductor_benchmarking.maybe_gpu_benchmark_lock():
                    time.sleep(0.001)
        results.put(("ok", os.getpid()))
    except BaseException as exc:
        results.put(("error", f"{type(exc).__name__}: {exc}"))


def test_results_payload_records_failures_without_changing_success_schema():
    successes = {
        "/tmp/ok/repro.py": {
            "default": {
                "compiled_us": 1.0,
                "memcopy_sol_us": 1.0,
                "gap_default": 1.0,
            }
        }
    }
    failures = {
        "/tmp/fail/repro.py": {
            "status": "failed",
            "gpu": "0",
            "elapsed": 0.1,
            "error": "intentional failure",
        }
    }

    payload = _results_payload(successes, failures, {"total": 2, "ok": 1, "failed": 1})

    assert payload["/tmp/ok/repro.py"] == successes["/tmp/ok/repro.py"]
    assert payload["__failures__"] == failures
    assert payload["__summary__"]["failed"] == 1


def test_default_worker_setup_does_not_take_gpu_lock():
    script = _persistent_worker_script("0", {
        "root": str(ROOT),
        "all_shapes": False,
        "no_cd": True,
        "n_warmup": 1,
        "n_rep": 1,
        "strict_gpu_lock": False,
    })

    assert "def gpu_setup_lock():" in script
    assert "STRICT_GPU_LOCK = False" in script
    assert "if STRICT_GPU_LOCK:\n        with gpu_bench_lock(\"shared\"):" in script
    assert "with gpu_bench_lock():\n                    return do_bench" in script


def test_strict_setup_lock_uses_inductor_lock_hook(monkeypatch, tmp_path):
    try:
        from torch._inductor.runtime import benchmarking as inductor_benchmarking
    except Exception as exc:
        pytest.skip(f"Inductor benchmarking unavailable: {exc}")

    if not hasattr(inductor_benchmarking, "set_gpu_benchmark_lock_context"):
        pytest.skip("Inductor benchmark lock hook unavailable")

    previous = inductor_benchmarking.set_gpu_benchmark_lock_context(None)
    script = _persistent_worker_script("0", {
        "root": str(ROOT),
        "all_shapes": False,
        "no_cd": True,
        "n_warmup": 1,
        "n_rep": 1,
        "strict_gpu_lock": True,
    })
    prefix = script.split("# Main loop:", 1)[0]
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("INDUCTOR_GPU_BENCH_LOCK", "1")
    monkeypatch.setenv("INDUCTOR_GPU_BENCH_LOCK_DIR", str(lock_dir))

    namespace = {"__name__": "bench_parallel_generated_worker_test"}
    try:
        exec(prefix, namespace)
        with namespace["gpu_setup_lock"]():
            lock_path = lock_dir / "gpu_0.lock"
            assert "mode=shared\n" in lock_path.read_text()
            with inductor_benchmarking.maybe_gpu_benchmark_lock():
                metadata = lock_path.read_text()
                assert "mode=exclusive\n" in metadata
                assert "label=bench_parallel_upgrade\n" in metadata
            assert "mode=shared\n" in lock_path.read_text()
    finally:
        inductor_benchmarking.set_gpu_benchmark_lock_context(previous)


def test_strict_setup_lock_stress_does_not_deadlock(tmp_path):
    try:
        from torch._inductor.runtime import benchmarking as inductor_benchmarking
    except Exception as exc:
        pytest.skip(f"Inductor benchmarking unavailable: {exc}")

    if not hasattr(inductor_benchmarking, "set_gpu_benchmark_lock_context"):
        pytest.skip("Inductor benchmark lock hook unavailable")
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("strict lock stress uses forked worker processes")

    script = _persistent_worker_script("0", {
        "root": str(ROOT),
        "all_shapes": False,
        "no_cd": True,
        "n_warmup": 1,
        "n_rep": 1,
        "strict_gpu_lock": True,
    })
    prefix = script.split("# Main loop:", 1)[0]
    ctx = mp.get_context("fork")
    workers = 4
    iterations = 25
    barrier = ctx.Barrier(workers)
    results = ctx.Queue()
    procs = [
        ctx.Process(
            target=_strict_lock_stress_worker,
            args=(prefix, str(tmp_path / "locks"), iterations, barrier, results),
        )
        for _ in range(workers)
    ]

    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=20)

    alive = [proc.pid for proc in procs if proc.is_alive()]
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)

    assert not alive, f"strict GPU lock stress deadlocked; live pids={alive}"
    statuses = [results.get_nowait() for _ in range(results.qsize())]
    errors = [payload for status, payload in statuses if status != "ok"]
    assert not errors
    assert len(statuses) == workers


def test_locked_worker_recovers_after_failures_and_releases_gpu_lock(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lock_dir = tmp_path / "locks"
        monkeypatch.setenv("INDUCTOR_GPU_BENCH_LOCK_DIR", str(lock_dir))

        task_queue = queue.Queue()
        repros = []
        for name in ("ok_before", "normal_fail", "cuda_fail", "ok_after"):
            repro = tmp_path / name / "repro.py"
            repro.parent.mkdir()
            repro.write_text("# fake repro\n")
            repros.append(repro)
            task_queue.put(repro)

        result_queue = queue.Queue()
        _locked_worker(
            {"index": "0", "name": "Fake GPU", "kind": "fake"},
            task_queue,
            result_queue,
            {
                "root": str(ROOT),
                "all_shapes": False,
                "no_cd": True,
                "n_warmup": 1,
                "n_rep": 1,
                "share_cache": False,
                "strict_gpu_lock": False,
                "n_workers": 1,
                "_persistent_worker_script_factory": _fake_persistent_worker_script,
            },
        )

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())

        assert [Path(result["repro"]).parent.name for result in results] == [
            "ok_before",
            "normal_fail",
            "cuda_fail",
            "ok_after",
        ]
        assert [result["status"] for result in results] == [
            "ok",
            "failed",
            "failed",
            "ok",
        ]
        assert "intentional non-cuda" in results[1]["error"]
        assert "CUDA_ERROR" in results[2]["error"]

        lock_path = lock_dir / "gpu_0.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping bench recovery smoke")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repros = [
            _write_repro(tmp_path / "ok_before" / "repro.py", VALID_REPRO),
            _write_repro(tmp_path / "normal_fail" / "repro.py", NORMAL_FAILURE_REPRO),
            _write_repro(tmp_path / "cuda_fail" / "repro.py", CUDA_FAILURE_REPRO),
            _write_repro(tmp_path / "ok_after" / "repro.py", VALID_REPRO),
        ]
        output = tmp_path / "results.json"
        env = os.environ.copy()
        env["INDUCTOR_GPU_BENCH_LOCK"] = "1"
        env["INDUCTOR_GPU_BENCH_LOCK_DIR"] = str(tmp_path / "locks")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor_cache")

        cmd = [
            sys.executable,
            "scripts/bench_parallel.py",
            *map(str, repros),
            "--gpus",
            os.environ.get("BB_TEST_GPU", "0"),
            "--max-workers",
            "1",
            "--workers-per-gpu",
            "1",
            "--no-share-cache",
            "--no-cd",
            "--n-warmup",
            "1",
            "--n-rep",
            "1",
            "--output",
            str(output),
        ]
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            return proc.returncode

        done_line = next(
            (line for line in reversed(proc.stdout.splitlines()) if line.startswith("Done:")),
            "",
        )
        assert "Done: 2 ok, 2 failed" in done_line, done_line

        results = json.loads(output.read_text())
        assert str(repros[0]) in results
        assert str(repros[3]) in results
        assert str(repros[1]) not in results
        assert str(repros[2]) not in results
        assert str(repros[1]) in results["__failures__"]
        assert str(repros[2]) in results["__failures__"]
        assert results["__summary__"]["ok"] == 2
        assert results["__summary__"]["failed"] == 2
        print("Worker recovery smoke passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
