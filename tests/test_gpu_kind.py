"""GPU kind detection: the process-visible CUDA device, newer kinds by name."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

import gpu_lock
import oracle_harness
import repro_harness
from gpu_kind import device_kind_from_name

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_device_kind_from_name_prefers_the_longest_kind():
    assert device_kind_from_name("NVIDIA GB300") == "GB300"
    assert device_kind_from_name("NVIDIA B300 SXM6 AC") == "B300"
    assert device_kind_from_name("NVIDIA B200") == "B200"
    assert device_kind_from_name("NVIDIA H100 80GB HBM3") == "H100"


def test_gpu_lock_reexports_the_shared_kind_helpers():
    assert gpu_lock.device_kind_from_name is device_kind_from_name


def test_kind_comes_from_the_process_visible_device(monkeypatch):
    names = {0: "NVIDIA B200", 1: "NVIDIA GB300"}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: names[index])
    cache = oracle_harness.get_gpu_kind.__dict__
    saved = cache.pop("_cached", None)
    try:
        assert repro_harness._detect_hardware() == "GB300"
        assert oracle_harness.get_gpu_kind() == "GB300"
    finally:
        cache.pop("_cached", None)
        if saved is not None:
            cache["_cached"] = saved


def test_kind_does_not_need_scripts_on_the_path(tmp_path):
    # An installed oracle (`pip install -e .`, then `python oracle.py`) has
    # only the packaged root modules importable, not scripts/.
    code = textwrap.dedent(
        f"""
        import sys
        sys.path[:] = [p for p in sys.path if p not in ("", {str(REPO_ROOT / "scripts")!r})]
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.modules["scripts"] = None
        sys.modules["scripts.gpu_lock"] = None
        sys.modules["gpu_lock"] = None
        import torch
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_name = lambda index: "NVIDIA H100 80GB HBM3"
        import oracle_harness, repro_harness
        print(oracle_harness.get_gpu_kind(), repro_harness._detect_hardware())
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["H100", "H100"]


def test_matching_gpus_matches_known_kinds_exactly(monkeypatch):
    devices = [
        {"index": "0", "name": "NVIDIA B200", "kind": "B200"},
        {"index": "1", "name": "NVIDIA GB300", "kind": "GB300"},
        {"index": "2", "name": "NVIDIA B300 SXM6 AC", "kind": "B300"},
    ]
    monkeypatch.setattr(gpu_lock, "discover_gpus", lambda: devices)
    assert [d["index"] for d in gpu_lock.matching_gpus("B300")] == ["2"]
    assert [d["index"] for d in gpu_lock.matching_gpus("gb300")] == ["1"]
    # An unlisted kind still matches by device-name substring.
    assert [d["index"] for d in gpu_lock.matching_gpus("SXM6")] == ["2"]
