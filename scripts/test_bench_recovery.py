"""Smoke-test bench_parallel worker recovery with intentionally bad repros."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


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
        print("Worker recovery smoke passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
