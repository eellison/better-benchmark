"""Pure-Python regression tests for the oracle --all-shapes SHARDING logic.

These tests exercise ONLY the pure helper functions in bench_parallel that
encode/split per-(dir,shape) task keys and regroup sharded results back into
the per-dir timings file model_graph_accounting consumes. They run with NO
GPU and do NOT import torch (verify: ``grep 'import torch' this_file`` is
empty), so they are safe to run while a production GPU sweep is in flight.

Why the stubs: two functions under test reach into modules that import torch
at module scope:
  - ``_aggregate_oracle_timings`` does ``from oracle_harness import
    _INVALID_STATUSES`` (oracle_harness imports torch).
  - ``_expand_oracle_shape_tasks`` does ``from repro_harness import
    load_shape_configs`` (repro_harness imports torch).
We install lightweight stand-ins in ``sys.modules`` BEFORE importing
bench_parallel, so the lazy imports resolve to the stubs and torch is never
pulled into the interpreter. ``_INVALID_STATUSES`` is copied verbatim from
oracle_harness.py so floor-exclusion behavior is faithful.

Run:  python scripts/test_oracle_sharding.py    (or: pytest scripts/test_oracle_sharding.py)
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Verbatim copy of oracle_harness._INVALID_STATUSES (oracle_harness.py).
_INVALID_STATUSES = frozenset({
    "UNVERIFIED_NUMERICS",
    "INVALID_CUDAGRAPH_WARNING",
    "NUMERICS_WORSE_THAN_COMPILED",
    "NO_ORACLE_FOR_SHAPE",
})

# File moved from scripts/ to tests/; bench_parallel lives in
# repo-root/scripts. conftest.py covers this for pytest; insert here too so
# `python tests/test_oracle_sharding.py` works standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_parallel as bp  # noqa: E402


# ── Torch-free stubs for the lazily-imported modules (SCOPED) ───────────────
# The two functions under test reach into modules that import torch at module
# scope, but only via *lazy* (in-function) imports fired at call time:
#   - ``_aggregate_oracle_timings`` -> ``from oracle_harness import _INVALID_STATUSES``
#   - ``_expand_oracle_shape_tasks`` -> ``from repro_harness import load_shape_configs``
# ``bench_parallel`` itself imports cleanly with NO torch, so we install the
# stubs only for the lifetime of this module's tests via an autouse fixture
# and RESTORE the original sys.modules entries on teardown. Installing them
# unconditionally at import time (the previous approach) leaked the stubs into
# the whole pytest session: any module collected afterwards that did e.g.
# ``from repro_harness import _eval_signature`` hit the stub and errored
# ``ImportError: cannot import name ...``. ``_INVALID_STATUSES`` is copied
# verbatim from oracle_harness.py so floor-exclusion behavior is faithful.
@pytest.fixture(autouse=True, scope="module")
def _torch_free_harness_stubs():
    sentinel = object()
    saved = {
        name: sys.modules.get(name, sentinel)
        for name in ("oracle_harness", "repro_harness")
    }

    _oh_stub = types.ModuleType("oracle_harness")
    _oh_stub._INVALID_STATUSES = _INVALID_STATUSES
    sys.modules["oracle_harness"] = _oh_stub

    _rh_stub = types.ModuleType("repro_harness")

    def _unset_load_shape_configs(*_a, **_k):  # pragma: no cover - replaced per test
        raise AssertionError(
            "load_shape_configs must be monkeypatched in the test before use"
        )

    _rh_stub.load_shape_configs = _unset_load_shape_configs
    sys.modules["repro_harness"] = _rh_stub

    try:
        yield
    finally:
        for name, prev in saved.items():
            if prev is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


# ── _make_shape_task_key / _split_shape_task_key round-trip ─────────────────

def test_task_key_roundtrip():
    key = bp._make_shape_task_key("repros/canonical/foo", "s16=24")
    assert key == "repros/canonical/foo::SHAPE::s16=24"
    assert bp._split_shape_task_key(key) == ("repros/canonical/foo", "s16=24")


def test_task_key_default_token_roundtrip():
    key = bp._make_shape_task_key("d/foo", bp._DEFAULT_SHAPE_TOKEN)
    assert bp._split_shape_task_key(key) == ("d/foo", bp._DEFAULT_SHAPE_TOKEN)


def test_split_bare_path_returns_none_label():
    # A plain dir path (no separator) → (path, None) so callers treat
    # un-sharded tasks transparently.
    assert bp._split_shape_task_key("repros/canonical/foo") == (
        "repros/canonical/foo",
        None,
    )


def test_split_rsplits_on_last_separator():
    # Label itself never contains the separator, but the dir path is arbitrary;
    # rsplit on the LAST sep keeps a weird path intact. (Defensive.)
    weird = "a::SHAPE::b"  # pathological dir containing the token
    key = bp._make_shape_task_key(weird, "lbl")
    assert bp._split_shape_task_key(key) == (weird, "lbl")


# ── _oracle_dir_name / _task_display_name suffix stripping ──────────────────

def test_oracle_dir_name_strips_shape_suffix():
    key = bp._make_shape_task_key("repros/canonical/myrepro", "s0=8")
    assert bp._oracle_dir_name(key) == "myrepro"
    # bare dir path
    assert bp._oracle_dir_name("repros/canonical/myrepro") == "myrepro"
    # repro.py file → parent dir name
    assert bp._oracle_dir_name("repros/canonical/myrepro/repro.py") == "myrepro"


def test_task_display_name_default_shows_bare_dir():
    key = bp._make_shape_task_key("repros/canonical/myrepro", bp._DEFAULT_SHAPE_TOKEN)
    assert bp._task_display_name(key) == "myrepro"


def test_task_display_name_nondefault_shows_dir_bracket_label():
    key = bp._make_shape_task_key("repros/canonical/myrepro", "s0=8")
    assert bp._task_display_name(key) == "myrepro[s0=8]"


# ── _expand_oracle_shape_tasks (load_shape_configs monkeypatched) ───────────

def test_expand_oracle_shape_tasks(monkeypatch):
    multi = Path("/x/repros/canonical/multi")
    single = Path("/x/repros/canonical/single")
    noconfig = Path("/x/repros/canonical/none")

    def fake_load(repro_file, symbol_bindings=None):
        # repro_file is "<dir>/repro.py"
        d = Path(repro_file).parent.name
        if d == "multi":
            return {"s0=8": object(), "s0=16": object(), "s0=32": object()}
        if d == "single":
            return {"only": object()}
        if d == "none":
            return {}  # falsy → __default__ task
        raise FileNotFoundError(repro_file)

    monkeypatch.setattr(sys.modules["repro_harness"], "load_shape_configs", fake_load)

    tasks = bp._expand_oracle_shape_tasks([multi, single, noconfig])
    assert tasks == [
        f"{multi}::SHAPE::s0=8",
        f"{multi}::SHAPE::s0=16",
        f"{multi}::SHAPE::s0=32",
        f"{single}::SHAPE::only",
        f"{noconfig}::SHAPE::__default__",
    ]


def test_expand_oracle_shape_tasks_load_raises_falls_back_to_default(monkeypatch):
    d = Path("/x/repros/canonical/boom")

    def boom(repro_file, symbol_bindings=None):
        raise RuntimeError("bad shapes.json")

    monkeypatch.setattr(sys.modules["repro_harness"], "load_shape_configs", boom)
    assert bp._expand_oracle_shape_tasks([d]) == [f"{d}::SHAPE::__default__"]


# ── _aggregate_oracle_timings: SHARDED vs UNSHARDED equivalence ─────────────

def _per_dir_payloads():
    """One payload per dir: {dir: {label: point, ...}}  (un-sharded shape)."""
    return {
        # 2 valid + 1 BAD_ORACLE (excluded from floor but kept in points)
        "/r/canonical/dirA/repro.py": {
            "modA_aaaa1111": {"status": "OK", "oracle_us": 10.0,
                              "compile_us": 20.0, "ratio": 2.0},
            "modA_bbbb2222": {"status": "OK", "oracle_us": 30.0,
                              "compile_us": 40.0, "ratio": 1.33},
            "modA_cccc3333": {"status": "BAD_ORACLE", "oracle_us": 99.0,
                              "compile_us": 50.0, "ratio": 0.5},
        },
        # only an invalid-status point → __failures__ no_valid_point
        "/r/canonical/dirB/repro.py": {
            "modB_dddd4444": {"status": "UNVERIFIED_NUMERICS", "oracle_us": 5.0,
                              "compile_us": 6.0, "ratio": 1.2},
        },
    }


def _per_shape_payloads():
    """One payload per (dir,shape): {dir::SHAPE::label: {label: point}}."""
    return {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=8"): {
            "modA_aaaa1111": {"status": "OK", "oracle_us": 10.0,
                              "compile_us": 20.0, "ratio": 2.0},
        },
        bp._make_shape_task_key("/r/canonical/dirA", "s0=16"): {
            "modA_bbbb2222": {"status": "OK", "oracle_us": 30.0,
                              "compile_us": 40.0, "ratio": 1.33},
        },
        bp._make_shape_task_key("/r/canonical/dirA", "s0=32"): {
            "modA_cccc3333": {"status": "BAD_ORACLE", "oracle_us": 99.0,
                              "compile_us": 50.0, "ratio": 0.5},
        },
        bp._make_shape_task_key("/r/canonical/dirB", "s0=8"): {
            "modB_dddd4444": {"status": "UNVERIFIED_NUMERICS", "oracle_us": 5.0,
                              "compile_us": 6.0, "ratio": 1.2},
        },
    }


def test_aggregate_sharded_equals_unsharded():
    unsharded = bp._aggregate_oracle_timings(_per_dir_payloads())
    sharded = bp._aggregate_oracle_timings(_per_shape_payloads())
    # The KEY equivalence: per-(dir,shape) payloads must aggregate IDENTICALLY
    # to one-payload-per-dir. (Bare assert of dict equality.)
    assert sharded == unsharded


def test_aggregate_shape_contents():
    # Lock in the actual aggregated content so the equivalence test isn't
    # trivially satisfied by two equal-but-wrong dicts.
    out = bp._aggregate_oracle_timings(_per_dir_payloads())
    # dirA priced from 2 valid points (BAD_ORACLE excluded from floor/median).
    assert out["dirA"]["n_points"] == 2
    assert out["dirA"]["oracle_us"] == 20.0   # median(10,30)
    assert out["dirA"]["compile_us"] == 30.0  # median(20,40)
    # BAD_ORACLE point is still recorded in points + points_by_shape.
    assert out["dirA"]["points"]["modA_cccc3333"]["status"] == "BAD_ORACLE"
    assert "cccc3333" in out["dirA"]["points_by_shape"]
    # dirB has no valid point → folded under __failures__ no_valid_point.
    assert "dirB" not in {k for k in out if k not in bp._RESERVED_TOP_LEVEL_KEYS}
    assert out["__failures__"]["dirB"]["reason"] == "no_valid_point"


# ── _regroup_sharded_oracle_failures ────────────────────────────────────────

def test_regroup_sharded_failures():
    failures = {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=8"): {"error": "boom-8"},
        bp._make_shape_task_key("/r/canonical/dirA", "s0=16"): {"error": "boom-16"},
        bp._make_shape_task_key("/r/canonical/dirB", "s0=4"): {"reason": "timeout"},
    }
    out = bp._regroup_sharded_oracle_failures(failures)
    assert set(out) == {"dirA", "dirB"}
    assert out["dirA"]["n_failed_shapes"] == 2
    assert sorted(out["dirA"]["failed_shapes"]) == ["s0=16", "s0=8"]
    # First-seen error preserved (dict insertion order → s0=8 boom-8).
    assert out["dirA"]["example_error"] == "boom-8"
    assert out["dirB"]["n_failed_shapes"] == 1
    assert out["dirB"]["failed_shapes"] == ["s0=4"]
    # reason used when no 'error' key.
    assert out["dirB"]["example_error"] == "timeout"


# ── _write_oracle_timings_output: FAILURE-ACCOUNTING matrix ─────────────────
# This is the critical, twice-fixed logic.

def _tmp_out():
    tmpdir = tempfile.mkdtemp(prefix="oracle_sharding_test_")
    # function does output_path.parent.mkdir — give it a nested Path, not a
    # bare mktemp string.
    return Path(tmpdir) / "nested" / "oracle_timings.json"


def _read_json(path: Path):
    return json.loads(path.read_text())


def test_write_case_a_partial_loss_still_priced():
    # (a) dir with 2 OK + 1 RAISED shape → dir PRICED (n_points==2) AND has
    # shape_failures (n_failed_shapes==1); no spurious __failures__.
    all_results = {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=8"): {
            "modA_aaaa1111": {"status": "OK", "oracle_us": 10.0,
                              "compile_us": 20.0, "ratio": 2.0},
        },
        bp._make_shape_task_key("/r/canonical/dirA", "s0=16"): {
            "modA_bbbb2222": {"status": "OK", "oracle_us": 30.0,
                              "compile_us": 40.0, "ratio": 1.33},
        },
    }
    failures = {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=32"): {"error": "OOM on s0=32"},
    }
    out_path = _tmp_out()
    timed = bp._write_oracle_timings_output(out_path, all_results, failures)

    assert timed["dirA"]["n_points"] == 2          # priced from surviving shapes
    sf = timed["dirA"]["shape_failures"]
    assert sf["n_failed_shapes"] == 1
    assert sf["failed_shapes"] == ["s0=32"]
    assert sf["example_error"] == "OOM on s0=32"
    assert "__failures__" not in timed             # no spurious failures key
    # return value matches what was written to disk
    assert _read_json(out_path) == timed


def test_write_case_b_no_valid_point_keeps_raised_shapes():
    # (b) THE BUG: dir with 1 UNVERIFIED_NUMERICS point (non-raised, in
    # all_results) + 2 RAISED shapes → dir in __failures__ reason
    # `no_valid_point` AND that entry carries shape_failures n_failed_shapes==2.
    # The raised shapes must NOT be dropped.
    all_results = {
        bp._make_shape_task_key("/r/canonical/dirB", "s0=8"): {
            "modB_dddd4444": {"status": "UNVERIFIED_NUMERICS", "oracle_us": 5.0,
                              "compile_us": 6.0, "ratio": 1.2},
        },
    }
    failures = {
        bp._make_shape_task_key("/r/canonical/dirB", "s0=16"): {"error": "raise-16"},
        bp._make_shape_task_key("/r/canonical/dirB", "s0=32"): {"error": "raise-32"},
    }
    out_path = _tmp_out()
    timed = bp._write_oracle_timings_output(out_path, all_results, failures)

    assert "dirB" not in {k for k in timed if k not in bp._RESERVED_TOP_LEVEL_KEYS}
    fail = timed["__failures__"]["dirB"]
    assert fail["reason"] == "no_valid_point"      # from the invalid point
    # raised shapes folded into the SAME entry, not dropped:
    sf = fail["shape_failures"]
    assert sf["n_failed_shapes"] == 2
    assert sorted(sf["failed_shapes"]) == ["s0=16", "s0=32"]
    assert _read_json(out_path) == timed


def test_write_case_c_all_shapes_raised():
    # (c) dir where the ONLY task(s) raised (nothing in all_results) →
    # __failures__ entry reason `all_shapes_failed`, n_failed_shapes correct.
    all_results = {}
    failures = {
        bp._make_shape_task_key("/r/canonical/dirC", "s0=8"): {"error": "raise-8"},
        bp._make_shape_task_key("/r/canonical/dirC", "s0=16"): {"error": "raise-16"},
    }
    out_path = _tmp_out()
    timed = bp._write_oracle_timings_output(out_path, all_results, failures)

    fail = timed["__failures__"]["dirC"]
    assert fail["reason"] == "all_shapes_failed"
    assert fail["n_failed_shapes"] == 2
    assert sorted(fail["failed_shapes"]) == ["s0=16", "s0=8"]
    assert _read_json(out_path) == timed


def test_write_case_d_no_failures_byte_identical_to_pure_aggregate():
    # (d) failures=None AND failures={} both produce output byte-identical to
    # pure _aggregate_oracle_timings, with NO spurious empty __failures__ key.
    all_results = {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=8"): {
            "modA_aaaa1111": {"status": "OK", "oracle_us": 10.0,
                              "compile_us": 20.0, "ratio": 2.0},
        },
        bp._make_shape_task_key("/r/canonical/dirA", "s0=16"): {
            "modA_bbbb2222": {"status": "OK", "oracle_us": 30.0,
                              "compile_us": 40.0, "ratio": 1.33},
        },
    }
    pure = bp._aggregate_oracle_timings(all_results)
    assert "__failures__" not in pure

    out_none = _tmp_out()
    timed_none = bp._write_oracle_timings_output(out_none, all_results, None)
    out_empty = _tmp_out()
    timed_empty = bp._write_oracle_timings_output(out_empty, all_results, {})

    assert timed_none == pure
    assert timed_empty == pure
    assert "__failures__" not in timed_none
    assert "__failures__" not in timed_empty
    # byte-identical on disk to a fresh dump of the pure aggregate
    assert out_none.read_bytes() == out_empty.read_bytes()


def test_write_case_e_aggregator_failures_survive_untouched():
    # (e) aggregator-produced __failures__ (no_valid_point from an invalid
    # point) survives untouched when there are NO worker failures.
    all_results = {
        bp._make_shape_task_key("/r/canonical/dirA", "s0=8"): {
            "modA_aaaa1111": {"status": "OK", "oracle_us": 10.0,
                              "compile_us": 20.0, "ratio": 2.0},
        },
        # dirB: only invalid point → aggregator records no_valid_point
        bp._make_shape_task_key("/r/canonical/dirB", "s0=8"): {
            "modB_dddd4444": {"status": "UNVERIFIED_NUMERICS", "oracle_us": 5.0,
                              "compile_us": 6.0, "ratio": 1.2},
        },
    }
    pure = bp._aggregate_oracle_timings(all_results)
    out_path = _tmp_out()
    for empty_failures in (None, {}):
        timed = bp._write_oracle_timings_output(out_path, all_results, empty_failures)
        assert timed == pure
        # dirB failure entry untouched (no shape_failures bolted on).
        assert timed["__failures__"]["dirB"]["reason"] == "no_valid_point"
        assert "shape_failures" not in timed["__failures__"]["dirB"]
        assert "dirA" in timed and timed["dirA"]["n_points"] == 1


# ── REPRO --all-shapes sharding: expansion + regroup ────────────────────────
# Mirrors the oracle sharding but for the plain repro path (no --oracles). The
# task unit becomes one (repro.py, shape) point; the parent regroups per-shape
# payloads back under the bare repro.py path so the output is byte-identical to
# the un-sharded in-worker-loop path.

def test_expand_repro_shape_tasks(monkeypatch):
    multi = Path("/x/repros/canonical/multi/repro.py")
    single = Path("/x/repros/canonical/single/repro.py")
    noconfig = Path("/x/repros/canonical/none/repro.py")

    def fake_load(repro_file, symbol_bindings=None):
        d = Path(repro_file).parent.name
        if d == "multi":
            return {"s0=8": object(), "s0=16": object()}
        if d == "single":
            return {"only": object()}
        if d == "none":
            return {}
        raise FileNotFoundError(repro_file)

    monkeypatch.setattr(sys.modules["repro_harness"], "load_shape_configs", fake_load)

    tasks = bp._expand_repro_shape_tasks([multi, single, noconfig])
    assert tasks == [
        f"{multi}::SHAPE::s0=8",
        f"{multi}::SHAPE::s0=16",
        f"{single}::SHAPE::only",
        f"{noconfig}::SHAPE::__default__",
    ]


def test_expand_repro_shape_tasks_load_raises_falls_back_to_default(monkeypatch):
    p = Path("/x/repros/canonical/boom/repro.py")

    def boom(repro_file, symbol_bindings=None):
        raise RuntimeError("bad shapes.json")

    monkeypatch.setattr(sys.modules["repro_harness"], "load_shape_configs", boom)
    assert bp._expand_repro_shape_tasks([p]) == [f"{p}::SHAPE::__default__"]


def test_sort_shape_tasks_big_dirs_first():
    # Two shapes of dirB + one of dirA + three of dirC -> dirC (3) then dirB (2)
    # then dirA (1); within a dir, original relative order is preserved by the
    # stable sort, and ties between equal-count dirs break on path.
    tasks = [
        bp._make_shape_task_key("/r/dirA/repro.py", "a0"),
        bp._make_shape_task_key("/r/dirB/repro.py", "b0"),
        bp._make_shape_task_key("/r/dirB/repro.py", "b1"),
        bp._make_shape_task_key("/r/dirC/repro.py", "c0"),
        bp._make_shape_task_key("/r/dirC/repro.py", "c1"),
        bp._make_shape_task_key("/r/dirC/repro.py", "c2"),
    ]
    out = bp._sort_shape_tasks_big_dirs_first(tasks)
    dirs_in_order = [bp._split_shape_task_key(t)[0] for t in out]
    # dirC (3 shapes) first, then dirB (2), then dirA (1)
    assert dirs_in_order == [
        "/r/dirC/repro.py", "/r/dirC/repro.py", "/r/dirC/repro.py",
        "/r/dirB/repro.py", "/r/dirB/repro.py",
        "/r/dirA/repro.py",
    ]


def _per_repro_shape_payloads():
    """Per-(repro, shape) payloads: {repro.py::SHAPE::label: {label: point}}."""
    a = "/r/canonical/dirA/repro.py"
    return {
        bp._make_shape_task_key(a, "s0=8"): {
            "s0=8": {"compiled_us": 10.0, "coord_descent_us": 9.0,
                     "memcopy_sol_us": 5.0, "total_bytes": 1024,
                     "gap_default": 2.0, "gap_cd": 1.8},
        },
        bp._make_shape_task_key(a, "s0=16"): {
            "s0=16": {"compiled_us": 20.0, "coord_descent_us": 18.0,
                      "memcopy_sol_us": 5.0, "total_bytes": 2048,
                      "gap_default": 4.0, "gap_cd": 3.6},
        },
    }


def _per_repro_unsharded_payload():
    """The equivalent un-sharded payload: one repro.py key, both shapes inside."""
    a = "/r/canonical/dirA/repro.py"
    return {
        a: {
            "s0=8": {"compiled_us": 10.0, "coord_descent_us": 9.0,
                     "memcopy_sol_us": 5.0, "total_bytes": 1024,
                     "gap_default": 2.0, "gap_cd": 1.8},
            "s0=16": {"compiled_us": 20.0, "coord_descent_us": 18.0,
                      "memcopy_sol_us": 5.0, "total_bytes": 2048,
                      "gap_default": 4.0, "gap_cd": 3.6},
        },
    }


def test_regroup_sharded_repro_results_equals_unsharded():
    # THE KEY equivalence: per-(repro, shape) payloads regroup IDENTICALLY to
    # the one-payload-per-repro (un-sharded in-worker-loop) output.
    regrouped = bp._regroup_sharded_repro_results(_per_repro_shape_payloads())
    assert regrouped == _per_repro_unsharded_payload()


def test_regroup_sharded_repro_results_bare_keys_passthrough():
    # Un-sharded (bare repro.py) keys pass through unchanged (idempotent).
    unsharded = _per_repro_unsharded_payload()
    assert bp._regroup_sharded_repro_results(unsharded) == unsharded


def test_regroup_repro_failures_all_shapes_failed():
    # Exact shape identity remains qualified even when every shape fails so a
    # focused merge cannot delete unselected baseline points.
    a = "/r/canonical/dirA/repro.py"
    all_results = {}  # nothing succeeded
    failures = {
        bp._make_shape_task_key(a, "s0=8"): {"error": "boom-8"},
        bp._make_shape_task_key(a, "s0=16"): {"error": "boom-16"},
    }
    out = bp._regroup_sharded_repro_failures(all_results, failures)
    assert set(out) == set(failures)
    assert out[bp._make_shape_task_key(a, "s0=8")]["error"] == "boom-8"
    assert out[bp._make_shape_task_key(a, "s0=16")]["error"] == "boom-16"


def test_regroup_repro_failures_partial_keeps_success_byte_identical():
    # Some shapes succeeded -> the success payload is left BYTE-IDENTICAL; the
    # failed shape is recorded under its SHAPE-qualified key (no collision).
    a = "/r/canonical/dirA/repro.py"
    all_results = bp._regroup_sharded_repro_results(_per_repro_shape_payloads())
    success_before = json.loads(json.dumps(all_results))  # deep copy snapshot
    failures = {
        bp._make_shape_task_key(a, "s0=32"): {"error": "boom-32"},
    }
    out = bp._regroup_sharded_repro_failures(all_results, failures)
    # success payload untouched
    assert all_results == success_before
    # the failed shape is accounted under its shape-qualified key, NOT the bare path
    assert a not in out
    assert bp._make_shape_task_key(a, "s0=32") in out
    assert out[bp._make_shape_task_key(a, "s0=32")]["error"] == "boom-32"


def test_regroup_repro_failures_bare_failure_passthrough():
    # An un-sharded (bare-path) failure passes through untouched.
    failures = {"/r/canonical/dirX/repro.py": {"error": "whole-repro-failed"}}
    out = bp._regroup_sharded_repro_failures({}, failures)
    assert out == failures


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
