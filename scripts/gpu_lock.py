"""Small file-lock helper for serializing GPU benchmarks.

This prevents multiple agents from benchmarking on the same GPU at the same
time.  The lock is advisory, so benchmark commands need to opt in by using this
helper or `scripts/with_gpu_lock.py`.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import time
from pathlib import Path
from typing import Iterator


DEFAULT_LOCK_DIR = Path(os.environ.get("COMPILE_UTILS_GPU_LOCK_DIR", "/tmp/compile_utils_gpu_locks"))


KNOWN_DEVICE_KINDS = ("B200", "H200", "H100", "A100", "V100")


def device_kind_from_name(name: str) -> str:
    for token in KNOWN_DEVICE_KINDS:
        if token.lower() in name.lower():
            return token
    return name.strip().replace(" ", "_") or "UNKNOWN"


def discover_gpus() -> list[dict[str, str]]:
    """Return physical GPU index/name/kind records from nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return [{"index": "0", "name": "UNKNOWN", "kind": "UNKNOWN"}]

    devices = []
    for line in out.splitlines():
        if not line.strip():
            continue
        index, name = [part.strip() for part in line.split(",", 1)]
        devices.append({"index": index, "name": name, "kind": device_kind_from_name(name)})
    return devices or [{"index": "0", "name": "UNKNOWN", "kind": "UNKNOWN"}]


def matching_gpus(device_kind: str | None = None) -> list[dict[str, str]]:
    devices = discover_gpus()
    if not device_kind:
        return devices
    wanted = device_kind.lower()
    return [
        device
        for device in devices
        if device["kind"].lower() == wanted or wanted in device["name"].lower()
    ]


def _write_lock_metadata(fd: int, *, gpu: str, label: str, device_kind: str = "") -> None:
    os.ftruncate(fd, 0)
    os.write(
        fd,
        (
            f"pid={os.getpid()}\n"
            f"gpu={gpu}\n"
            f"device_kind={device_kind}\n"
            f"label={label}\n"
            f"acquired_unix={time.time():.0f}\n"
        ).encode(),
    )
    os.fsync(fd)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_lock_pid(fd: int) -> int | None:
    """Read the PID from a lock file's metadata."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        content = os.read(fd, 256).decode()
        for line in content.splitlines():
            if line.startswith("pid="):
                return int(line.split("=")[1])
    except Exception:
        pass
    return None


@contextlib.contextmanager
def gpu_lock(
    gpu: int | str = 0,
    *,
    lock_dir: Path | str | None = None,
    timeout_s: float | None = None,
    label: str = "",
) -> Iterator[Path]:
    """Acquire an exclusive advisory lock for one GPU.

    Handles stale locks: if the lock is held by a dead process, it's
    forcibly released (the flock is automatically dropped when the holding
    process dies, but the file may still exist with stale metadata).
    """
    directory = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"gpu_{gpu}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            # Check if the holder is still alive
            holder_pid = _read_lock_pid(fd)
            if holder_pid is not None and not _is_pid_alive(holder_pid):
                # Stale lock — force acquire (the flock should already be released
                # by the kernel when the process died, so try again immediately)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    pass
            if timeout_s is not None and time.monotonic() - start >= timeout_s:
                os.close(fd)
                raise TimeoutError(f"Timed out waiting for GPU {gpu} lock: {lock_path}")
            time.sleep(1.0)

    try:
        _write_lock_metadata(fd, gpu=str(gpu), label=label)
        yield lock_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def gpu_lock_for_kind(
    device_kind: str | None = None,
    *,
    lock_dir: Path | str | None = None,
    timeout_s: float | None = None,
    label: str = "",
) -> Iterator[dict[str, str]]:
    """Acquire any free GPU lock matching a device kind.

    Yields a device record with `index`, `name`, `kind`, and `lock_path`.
    """
    directory = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()

    while True:
        devices = matching_gpus(device_kind)
        if not devices:
            available = ", ".join(
                f"{d['index']}:{d['kind']}:{d['name']}" for d in discover_gpus()
            )
            raise RuntimeError(
                f"No GPUs match device kind {device_kind!r}. Available: {available}"
            )

        for device in devices:
            lock_path = directory / f"gpu_{device['index']}.lock"
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue

            try:
                _write_lock_metadata(
                    fd,
                    gpu=device["index"],
                    device_kind=device["kind"],
                    label=label,
                )
                yield {**device, "lock_path": str(lock_path)}
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            return

        if timeout_s is not None and time.monotonic() - start >= timeout_s:
            raise TimeoutError(f"Timed out waiting for GPU kind {device_kind!r}")
        time.sleep(1.0)


@contextlib.contextmanager
def gpu_shared_lock(
    gpu: int | str = 0,
    *,
    lock_dir: Path | str | None = None,
    timeout_s: float | None = None,
    label: str = "",
) -> Iterator[Path]:
    """Acquire a shared (non-exclusive) advisory lock for one GPU.

    Multiple processes can hold shared locks simultaneously (e.g., for
    CUDA initialization, tensor allocation, graph lowering).

    An exclusive lock (gpu_lock) will block until all shared locks are
    released — use exclusive for ANY benchmarking (both autotuning's
    do_bench and our final measurement).

    Shared lock use cases:
      - torch.randn(device='cuda'), tensor creation
      - CUDA context init
      - Graph lowering / Triton codegen (CPU-bound but may touch GPU for shape queries)

    Exclusive lock use cases:
      - Inductor autotuning (benchmarker.benchmark / do_bench)
      - Our final do_bench measurement
      - Any timing-sensitive GPU operation
    """
    directory = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"gpu_{gpu}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if timeout_s is not None and time.monotonic() - start >= timeout_s:
                os.close(fd)
                raise TimeoutError(f"Timed out waiting for shared GPU {gpu} lock: {lock_path}")
            time.sleep(0.1)

    try:
        yield lock_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
