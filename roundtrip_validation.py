"""Round-trip invariant validation for saved full_graph_*.py artifacts.

Three invariants (driven by scripts/validate_corpus_invariants.py
--full-graph-roundtrip and the capture-time validate-before-write gate):

  A (input round-trip): the saved artifact + sidecar reconstructs runnable
    inputs via full_graph_harness (load_full_graph_definition +
    make_tensor_from_spec). Checked under FakeTensorMode so no GPU memory is
    touched. Failures name the offending placeholder / tensor attribute.

  B (partition determinism): capture_hook.get_fusion_partitions run twice on
    the same traced GraphModule yields identical pattern-hash multisets.

  C (partition round-trip): the pattern-hash set derived by re-tracing the
    saved artifact matches the recorded patterns (manifest.json patterns
    offline, or the live capture partitions in the validate-before-write
    gate). Mismatches caused purely by view-op spelling drift during
    re-tracing (aten.reshape -> aten.view, aten.t -> aten.permute, ...) are
    classified as a KNOWN class ("spelling"); everything else is a hard
    mismatch.

NOTE on multiset vs set for C: manifest.json stores each pattern hash once
(deduplicated), while a model graph legitimately contains the same pattern
many times, so C compares hash SETS. B compares true multisets (same graph,
same partitioner, run twice).

This module intentionally does NOT reimplement any hashing: it always calls
capture_hook.get_fusion_partitions / compute_partition_pattern so it stays in
lockstep with the capture pipeline's single source of truth.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

_OP_RE = re.compile(r"torch\.ops\.((?:\w+\.)+\w+)")


# ---------------------------------------------------------------------------
# Fake-mode context (no GPU work, runs on CUDA-less machines)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def fake_tensor_ctx():
    """FakeTensorMode that can fabricate cuda tensors even without a GPU.

    All round-trip validation runs under this context: input construction and
    make_fx tracing never allocate real device memory.
    """
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    if torch.cuda.is_available():
        with mode:
            yield mode
    else:
        from unittest.mock import patch

        with mode, patch.object(torch.cuda, "is_available", return_value=True):
            yield mode


# ---------------------------------------------------------------------------
# Invariant A: input round-trip
# ---------------------------------------------------------------------------

def check_input_roundtrip(definition) -> list[str]:
    """Invariant A: every input spec / tensor attr must be constructible.

    Returns a list of failure strings, each naming the placeholder (or
    tensor attribute) and the reason. Empty list == pass.
    """
    from full_graph_harness import make_tensor_from_spec

    failures: list[str] = []
    if not definition.input_specs and not definition.forward_takes_no_inputs:
        failures.append(
            "no parseable forward inputs (annotations and sidecar both empty)"
        )
        return failures

    with fake_tensor_ctx():
        for idx, spec in enumerate(definition.input_specs):
            name = spec.get("name") or f"input_{idx}"
            if spec.get("kind") != "tensor":
                continue  # symint/scalar specs cannot fail construction
            try:
                make_tensor_from_spec(spec, default_device="cuda")
            except Exception as exc:
                failures.append(
                    f"placeholder {name!r}: {type(exc).__name__}: {exc}"
                )
        for attr_name, spec in (definition.tensor_attrs or {}).items():
            try:
                make_tensor_from_spec(spec, default_device="cuda")
            except Exception as exc:
                failures.append(
                    f"tensor_attr {attr_name!r}: {type(exc).__name__}: {exc}"
                )
    return failures


# ---------------------------------------------------------------------------
# Tracing + partition derivation (delegates to capture_hook helpers)
# ---------------------------------------------------------------------------

def trace_full_graph_fake(definition):
    """Re-trace a loaded full-graph definition with make_fx under fake mode."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    from full_graph_harness import (
        instantiate_full_graph,
        make_inputs_from_full_graph_specs,
    )

    with fake_tensor_ctx():
        instance = instantiate_full_graph(definition, default_device="cuda")
        inputs = make_inputs_from_full_graph_specs(
            definition.input_specs, default_device="cuda"
        )
        with torch.no_grad():
            return make_fx(instance, tracing_mode="fake")(*inputs)


def derive_partitions(gm) -> list[dict[str, Any]]:
    """Partition a GraphModule with the capture pipeline's own partitioner.

    Returns a list of {"hash": pattern_hash, "ops": [str(target), ...]} —
    one entry per fusible partition occurrence.
    """
    from capture_hook import compute_partition_pattern, get_fusion_partitions

    out = []
    for comp in get_fusion_partitions(gm):
        pattern = compute_partition_pattern(comp, gm)
        if pattern is None:
            continue
        out.append({
            "hash": pattern["pattern_hash"],
            "ops": sorted(
                str(n.target) for n in comp if n.op == "call_function"
            ),
        })
    return out


def partition_hash_multiset(parts: list[dict[str, Any]]) -> Counter:
    return Counter(p["hash"] for p in parts)


# ---------------------------------------------------------------------------
# Invariant B: partition determinism
# ---------------------------------------------------------------------------

def check_partition_determinism(gm) -> tuple[bool, str | None]:
    """Invariant B: two partitioner runs on the same gm -> same hash multiset.

    Returns (ok, diff_description). diff_description names the differing
    hashes (with multiplicities) when ok is False.
    """
    first = partition_hash_multiset(derive_partitions(gm))
    second = partition_hash_multiset(derive_partitions(gm))
    if first == second:
        return True, None
    only_first = dict(first - second)
    only_second = dict(second - first)
    return False, (
        f"hash multisets differ between runs: "
        f"run1-only={only_first} run2-only={only_second}"
    )


# ---------------------------------------------------------------------------
# Invariant C: partition round-trip (with spelling-equivalence classification)
# ---------------------------------------------------------------------------

def _transparent_op_names() -> set[str]:
    from capture_hook import TRANSPARENT_OPS

    return {str(op) for op in TRANSPARENT_OPS}


def nontransparent_key(ops: list[str]) -> tuple[str, ...]:
    """Multiset of a partition's ops with transparent view/metadata ops
    removed — re-tracing only re-spells those (reshape->view, t->permute,
    getitem placement), so two partitions with the same non-transparent key
    describe the same compute."""
    transparent = _transparent_op_names()
    return tuple(sorted(op for op in ops if op not in transparent))


def canonical_repro_ops(
    pattern_hash: str, canonical_dir: str | Path
) -> list[str] | None:
    """Op list of the canonical repro for a pattern hash (regex over repro.py),
    in the same str(node.target) spelling ("aten.relu.default")."""
    for d in sorted(Path(canonical_dir).glob(f"*_{pattern_hash}")):
        repro = d / "repro.py"
        if repro.exists():
            return sorted(_OP_RE.findall(repro.read_text()))
    return None


def compare_pattern_sets(
    derived_parts: list[dict[str, Any]],
    expected_hashes,
    *,
    expected_ops_lookup: Callable[[str], list[str] | None],
) -> dict[str, Any]:
    """Invariant C: derived pattern-hash set vs expected pattern-hash set.

    expected_ops_lookup(hash) -> op list (or None if unknown) is used to
    classify each mismatch:
      - spelling: the missing/extra partition has the same non-transparent op
        multiset as a partition on the other side (re-trace view-op spelling
        drift, e.g. aten.reshape.default -> aten.view.default). KNOWN class.
      - hard: no such counterpart — a real round-trip violation.

    Returns {"ok", "missing", "extra", "spelling", "hard"} where ok means
    no hard mismatches.
    """
    derived_hashes = {p["hash"] for p in derived_parts}
    expected = set(expected_hashes)
    missing = sorted(expected - derived_hashes)
    extra = sorted(derived_hashes - expected)

    derived_keys = {nontransparent_key(p["ops"]) for p in derived_parts}
    expected_keys: set[tuple[str, ...]] = set()
    for h in expected:
        ops = expected_ops_lookup(h)
        if ops is not None:
            expected_keys.add(nontransparent_key(ops))

    spelling: list[str] = []
    hard: list[str] = []

    for h in missing:
        ops = expected_ops_lookup(h)
        if ops is not None and nontransparent_key(ops) in derived_keys:
            spelling.append(
                f"missing {h} (expected ops {ops} re-traced under a different "
                f"view-op spelling)"
            )
        elif ops is None:
            hard.append(f"missing {h} (no op info available to classify)")
        else:
            hard.append(f"missing {h} (expected ops {ops}, no derived "
                        f"partition with matching compute)")

    extra_ops_by_hash: dict[str, list[str]] = {}
    for p in derived_parts:
        extra_ops_by_hash.setdefault(p["hash"], p["ops"])
    for h in extra:
        ops = extra_ops_by_hash.get(h, [])
        if nontransparent_key(ops) in expected_keys:
            spelling.append(
                f"extra {h} (derived ops {ops} match an expected partition "
                f"under a different view-op spelling)"
            )
        else:
            hard.append(f"extra {h} (derived ops {ops} not in expected set)")

    return {
        "ok": not hard,
        "missing": missing,
        "extra": extra,
        "spelling": spelling,
        "hard": hard,
    }


# ---------------------------------------------------------------------------
# Validate-before-write gate (used by capture_hook / extract_reductions)
# ---------------------------------------------------------------------------

def validate_written_full_graph(
    graph_path: str | Path,
    *,
    original_gm=None,
) -> dict[str, Any]:
    """Run invariants A + C against a just-written full_graph_*.py artifact.

    original_gm: the live post-grad GraphModule the artifact was written
    from. When provided, C compares the re-traced artifact's pattern-hash set
    against the live gm's partitions (op lists taken directly from the live
    partitions for spelling classification).

    Never raises. Returns:
        {"status": "ok" | "failed",
         "reason": str | None,          # first failure, for the sidecar stamp
         "a_failures": [...],
         "c": {...} | None}
    """
    result: dict[str, Any] = {
        "status": "ok",
        "reason": None,
        "a_failures": [],
        "c": None,
    }

    def _fail(reason: str):
        if result["status"] == "ok":
            result["status"] = "failed"
            result["reason"] = reason

    # --- Invariant A ---
    try:
        from full_graph_harness import load_full_graph_definition

        definition = load_full_graph_definition(graph_path)
    except Exception as exc:
        _fail(f"A: load_full_graph_definition: {type(exc).__name__}: {exc}")
        return result

    try:
        a_failures = check_input_roundtrip(definition)
    except Exception as exc:
        a_failures = [f"input round-trip crashed: {type(exc).__name__}: {exc}"]
    result["a_failures"] = a_failures
    if a_failures:
        _fail(f"A: {a_failures[0]}")
        return result

    # --- Invariant C (needs the original gm to compare against) ---
    if original_gm is None:
        return result
    try:
        original_parts = derive_partitions(original_gm)
        gm = trace_full_graph_fake(definition)
        derived_parts = derive_partitions(gm)
    except Exception as exc:
        _fail(f"C: re-trace failed: {type(exc).__name__}: {exc}")
        return result

    ops_by_hash: dict[str, list[str]] = {}
    for p in original_parts:
        ops_by_hash.setdefault(p["hash"], p["ops"])
    comparison = compare_pattern_sets(
        derived_parts,
        ops_by_hash.keys(),
        expected_ops_lookup=ops_by_hash.get,
    )
    result["c"] = comparison
    if not comparison["ok"]:
        _fail(f"C: {comparison['hard'][0]}")
    return result


def stamp_roundtrip(graph_path: str | Path, verdict: dict[str, Any]) -> None:
    """Stamp the .meta.json sidecar with the round-trip verdict.

    Writes "roundtrip": "ok" | "failed: <reason>" (plus the spelling-class
    notes when present). Creates a minimal sidecar if none exists. Never
    deletes the artifact; failures additionally print a loud stderr WARNING.
    """
    graph_path = Path(graph_path)
    meta_path = graph_path.with_suffix(".meta.json")
    try:
        sidecar = (
            json.loads(meta_path.read_text()) if meta_path.exists() else {}
        )
        if not isinstance(sidecar, dict):
            sidecar = {}
    except Exception:
        sidecar = {}
    sidecar.setdefault("graph", graph_path.name)

    if verdict.get("status") == "ok":
        sidecar["roundtrip"] = "ok"
    else:
        sidecar["roundtrip"] = f"failed: {verdict.get('reason')}"
        print(
            f"[roundtrip] WARNING: {graph_path} failed round-trip validation: "
            f"{verdict.get('reason')} (artifact kept, sidecar stamped)",
            file=sys.stderr,
        )
    c = verdict.get("c") or {}
    if c.get("spelling"):
        sidecar["roundtrip_spelling_notes"] = c["spelling"]

    try:
        meta_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    except Exception as exc:
        print(
            f"[roundtrip] WARNING: could not stamp sidecar {meta_path}: {exc}",
            file=sys.stderr,
        )


def run_write_gate(graph_path: str | Path, original_gm=None) -> dict[str, Any]:
    """Validate-before-write gate for capture paths that emit full_graph_*.py.

    Call right after the artifact (+ sidecar) is written. Reloads the
    just-written artifact, runs invariants A + C against the live gm, and
    stamps the sidecar with "roundtrip": "ok" | "failed: <reason>". On
    failure prints a loud stderr WARNING but NEVER deletes the artifact.

    Runs outside any ambient FakeTensorMode (capture hooks fire inside
    Inductor compilation, where a fake mode may be active) and never raises.
    """
    graph_path = Path(graph_path)
    try:
        try:
            from torch._subclasses.fake_tensor import unset_fake_temporarily
        except Exception:
            unset_fake_temporarily = contextlib.nullcontext  # type: ignore

        with unset_fake_temporarily():
            verdict = validate_written_full_graph(
                graph_path, original_gm=original_gm
            )
    except Exception as exc:
        verdict = {
            "status": "failed",
            "reason": f"round-trip gate crashed: {type(exc).__name__}: {exc}",
            "a_failures": [],
            "c": None,
        }
    stamp_roundtrip(graph_path, verdict)
    return verdict
