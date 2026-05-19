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
STALE_LOCK_GRACE_S = float(os.environ.get("COMPILE_UTILS_GPU_LOCK_STALE_GRACE_S", "5"))


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


def _write_lock_metadata(
    fd: int,
    *,
    gpu: str,
    label: str,
    device_kind: str = "",
    mode: str = "exclusive",
) -> None:
    os.ftruncate(fd, 0)
    os.write(
        fd,
        (
            f"pid={os.getpid()}\n"
            f"gpu={gpu}\n"
            f"device_kind={device_kind}\n"
            f"mode={mode}\n"
            f"label={label}\n"
            f"acquired_unix={time.time():.0f}\n"
        ).encode(),
    )
    os.fsync(fd)


def _read_lock_metadata(lock_path: Path) -> dict[str, str]:
    try:
        text = lock_path.read_text()
    except OSError:
        return {}

    metadata = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata


def _pid_is_alive(pid: str | None) -> bool | None:
    if not pid:
        return None
    try:
        pid_int = int(pid)
    except ValueError:
        return None
    if pid_int <= 0:
        return False

    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _format_lock_holder(metadata: dict[str, str]) -> str:
    if not metadata:
        return "unknown holder"

    fields = [
        f"{key}={metadata[key]}"
        for key in ("pid", "gpu", "device_kind", "mode", "label", "acquired_unix")
        if metadata.get(key)
    ]
    return ", ".join(fields) if fields else "unknown holder"


def _note_stale_holder_or_raise(
    *,
    lock_path: Path,
    stale_since: float | None,
    now: float,
) -> float | None:
    metadata = _read_lock_metadata(lock_path)
    if _pid_is_alive(metadata.get("pid")) is not False:
        return None

    stale_since = stale_since if stale_since is not None else now
    if now - stale_since >= STALE_LOCK_GRACE_S:
        raise TimeoutError(
            f"GPU lock appears stale: {lock_path} reports dead "
            f"{_format_lock_holder(metadata)}"
        )
    return stale_since


def _timeout_error(lock_path: Path, description: str) -> TimeoutError:
    metadata = _read_lock_metadata(lock_path)
    return TimeoutError(
        f"Timed out waiting for {description}: {lock_path} "
        f"({_format_lock_holder(metadata)})"
    )


@contextlib.contextmanager
def gpu_lock(
    gpu: int | str = 0,
    *,
    lock_dir: Path | str | None = None,
    timeout_s: float | None = None,
    label: str = "",
) -> Iterator[Path]:
    """Acquire an exclusive advisory lock for one GPU."""
    directory = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"gpu_{gpu}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    start = time.monotonic()
    stale_since = None

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            now = time.monotonic()
            try:
                stale_since = _note_stale_holder_or_raise(
                    lock_path=lock_path,
                    stale_since=stale_since,
                    now=now,
                )
            except TimeoutError:
                os.close(fd)
                raise
            if timeout_s is not None and now - start >= timeout_s:
                os.close(fd)
                raise _timeout_error(lock_path, f"GPU {gpu} lock")
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
    stale_since_by_path: dict[Path, float | None] = {}

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
                now = time.monotonic()
                try:
                    stale_since_by_path[lock_path] = _note_stale_holder_or_raise(
                        lock_path=lock_path,
                        stale_since=stale_since_by_path.get(lock_path),
                        now=now,
                    )
                except TimeoutError:
                    os.close(fd)
                    raise
                os.close(fd)
                continue

            try:
                _write_lock_metadata(
                    fd,
                    gpu=device["index"],
                    device_kind=device["kind"],
                    label=label,
                    mode="exclusive",
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
    stale_since = None

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            now = time.monotonic()
            try:
                stale_since = _note_stale_holder_or_raise(
                    lock_path=lock_path,
                    stale_since=stale_since,
                    now=now,
                )
            except TimeoutError:
                os.close(fd)
                raise
            if timeout_s is not None and now - start >= timeout_s:
                os.close(fd)
                raise _timeout_error(lock_path, f"shared GPU {gpu} lock")
            time.sleep(0.1)

    try:
        _write_lock_metadata(fd, gpu=str(gpu), label=label, mode="shared")
        yield lock_path
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
