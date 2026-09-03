"""End-to-end test for the --worker-init flag."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


REPRO = """
import torch


class Repro(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + 1.0


def make_inputs():
    return (torch.randn((16,), device="cuda"),)
"""

# The init records the pid it ran in. A worker is a separate process, so the
# pid is what tells apart "ran where it had to" from "ran in the parent".
INIT_MODULE = """
import os


def install():
    with open(os.environ["WORKER_INIT_MARKER"], "a") as fh:
        fh.write(f"{os.getpid()}\\n")
"""


def test_worker_init_runs_in_the_worker(tmp_path):
    """Run the real CLI with --worker-init and check the callable fired.

    Not asserted through stdout: a worker redirects fd 1 to stderr and reports
    results on a dedicated fd, and the parent only surfaces worker stderr when
    the repro fails. So the init leaves a marker file behind instead.
    """
    if not torch.cuda.is_available():
        pytest.skip("GPU required to run a repro end to end")

    repro = tmp_path / "sin" / "repro.py"
    repro.parent.mkdir()
    repro.write_text(REPRO)

    (tmp_path / "init_probe.py").write_text(INIT_MODULE)
    marker = tmp_path / "fired.txt"
    output = tmp_path / "results.json"

    env = os.environ.copy()
    env["WORKER_INIT_MARKER"] = str(marker)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor_cache")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bench_parallel.py",
            str(repro),
            "--gpus", os.environ.get("BB_TEST_GPU", "0"),
            "--max-workers", "1",
            "--workers-per-gpu", "1",
            "--no-share-cache",
            "--no-cd",
            "--n-warmup", "1",
            "--n-rep", "1",
            "--worker-init", "init_probe:install",
            "--output", str(output),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )

    assert proc.returncode == 0, proc.stdout
    assert "Done: 1 ok, 0 failed" in proc.stdout, proc.stdout
    # The parent says it is going to do this...
    assert "Worker init: init_probe:install" in proc.stdout, proc.stdout

    # ...and this is it having been done, in a process that is not the parent.
    assert marker.exists(), proc.stdout
    pids = {int(line) for line in marker.read_text().split()}
    assert pids and os.getpid() not in pids

    results = json.loads(output.read_text())
    assert results["_metadata"]["benchmark_config"]["worker_init"] == [
        "init_probe:install"
    ]
