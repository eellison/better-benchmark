"""Guard test: NO in-process GPU timing without the GPU lock.

An unlocked GPU timing window is INVALID, not merely noisier — a co-tenant
kernel on the same device inflates the measured time arbitrarily. This is not
hypothetical: the genai microbenchmark "±10-44% per-commit swing" that polluted
a perf-attribution series was traced (2026-06-23) to exactly this — a bespoke
timer (`scripts/genai_single.py`, since deleted) that called ``do_bench``
without acquiring the lock. The canonical locked path measures the SAME kernels
at 0.03-0.52% spread. A second instance recurred during the mega-commit
ablation: genai off-arms run 3-concurrent on separate GPUs showed bogus 3-19%
"wins" until re-run in isolation under the lock.

The invariant this test enforces:

    Any file that times with ``do_bench`` (or raw CUDA-event timing) MUST either
      (a) acquire ``scripts/gpu_lock.gpu_lock`` itself, OR
      (b) delegate timing to the locked harness (repro_harness.benchmark_repro /
          oracle_harness.bench_oracle / bench_parallel, which set
          ``INDUCTOR_GPU_BENCH_LOCK`` and flock per GPU).

Files that contain ``do_bench`` only inside a *string literal* (e.g.
``extract_reductions.py`` emits a benchmark template into generated repros — it
never times anything itself) are naturally exempt: we parse with ``ast`` and
only flag *real call sites*, never substring matches.

This is the static-analysis backstop for the "never write new benches, reuse"
rule. It runs with NO GPU and does NOT import torch, so it is safe in CI and
while a production sweep is in flight.

Run:  pytest tests/test_no_unlocked_timing.py    (or: python tests/test_no_unlocked_timing.py)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories we do NOT scan: tests (this guard + fixtures may name do_bench),
# generated/scratch output, vendored caches, and the venv.
_SKIP_DIRS = {
    "tests",
    "results",
    "investigation_results",
    ".claude",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "benchmarks",
    "repros",  # generated repro modules carry an emitted bench template
}

# Markers that prove a file participates in the locked timing path. Presence of
# ANY of these (as a substring of the source) exempts the file: it either calls
# gpu_lock directly, or routes timing through a known-locked harness entrypoint.
_LOCK_MARKERS = (
    "gpu_lock",                 # scripts/gpu_lock.py context managers
    "gpu_lock_for_kind",
    "INDUCTOR_GPU_BENCH_LOCK",  # env flag the locked worker sets
    "benchmark_repro",          # repro_harness locked entrypoint
    "bench_oracle",             # oracle_harness locked entrypoint
)

# Function names that constitute "an in-process GPU timing call". do_bench is
# the only one used in this repo today; the CUDA-event names are included so a
# future hand-rolled timer is also caught.
_TIMING_CALLS = {"do_bench"}
_TIMING_ATTRS = {"do_bench", "elapsed_time"}

# Explicit, justified exemptions. MUST stay empty in normal operation; adding an
# entry requires a comment explaining why the file legitimately times without a
# lock (there is no such legitimate case for GPU timing — prefer fixing the file).
_ALLOWLIST: set[str] = set()


def _python_files() -> list[Path]:
    out = []
    for p in REPO_ROOT.rglob("*.py"):
        rel_parts = p.relative_to(REPO_ROOT).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        out.append(p)
    return out


def _real_timing_callsites(tree: ast.AST) -> list[str]:
    """Return descriptions of real do_bench/elapsed_time CALL nodes.

    Uses the AST, so ``do_bench`` appearing only inside a string literal (a
    template emitted into generated code) is NOT flagged.
    """
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in _TIMING_CALLS:
            hits.append(f"{fn.id}() @ line {node.lineno}")
        elif isinstance(fn, ast.Attribute) and fn.attr in _TIMING_ATTRS:
            hits.append(f".{fn.attr}() @ line {node.lineno}")
    return hits


def test_no_unlocked_do_bench():
    """Every real do_bench/elapsed_time call site sits in a lock-aware file."""
    offenders: dict[str, list[str]] = {}

    for path in _python_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in _ALLOWLIST:
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        # Cheap pre-filter: no "do_bench"/"elapsed_time" substring at all -> skip parse.
        if "do_bench" not in src and "elapsed_time" not in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # not our concern here
        callsites = _real_timing_callsites(tree)
        if not callsites:
            continue  # only in strings/comments -> exempt (e.g. emitted templates)
        if any(marker in src for marker in _LOCK_MARKERS):
            continue  # file participates in the locked path
        offenders[rel] = callsites

    assert not offenders, (
        "Unlocked GPU timing detected — these files call do_bench/elapsed_time "
        "without acquiring the GPU lock or delegating to the locked harness. An "
        "unlocked timing window is INVALID (co-tenant kernels inflate it). Wrap "
        "the timed region in `from gpu_lock import gpu_lock` (acquire before the "
        "torch/CUDA work) and use return_mode='min', OR drive bench_parallel / "
        "benchmark_repro / bench_oracle. See this file's docstring for the "
        "genai ±10-44% incident.\n"
        + "\n".join(f"  {f}: {', '.join(sites)}" for f, sites in sorted(offenders.items()))
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
