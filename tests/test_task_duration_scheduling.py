import json
from pathlib import Path

import pytest

from scripts.bench_parallel import (
    _REPO_ROOT,
    _load_task_duration_ledger,
    _record_task_duration,
    _sort_tasks_by_known_duration,
    _task_duration_key,
    _update_task_duration_ledger,
)


def _ledger(entries=None):
    return {"schema_version": 1, "tasks": dict(entries or {})}


def test_task_duration_key_is_stable_across_external_corpus_roots():
    first = Path("/tmp/corpus-a/models/hf/infer/model/full_graph_001.py")
    second = Path("/mnt/corpus-b/models/hf/infer/model/full_graph_001.py")

    assert _task_duration_key(first, "full_graph", "B200") == _task_duration_key(
        second, "full_graph", "B200"
    )
    assert _task_duration_key(first, "full_graph", "B200") == (
        "B200:full_graph:models/hf/infer/model/full_graph_001.py"
    )


def test_task_duration_key_separates_hardware_kinds():
    task = Path("/tmp/corpus/canonical/foo/repro.py")
    assert _task_duration_key(task, "repro", "B200") != _task_duration_key(
        task, "repro", "GB300"
    )


def test_task_duration_key_preserves_shape_identity():
    task = "/tmp/corpus/canonical/foo/repro.py::SHAPE::batch_32"

    assert _task_duration_key(task, "repro", "B200") == (
        "B200:repro:canonical/foo/repro.py::SHAPE::batch_32"
    )


def test_known_tasks_sort_longest_first_and_unknowns_remain_stable():
    tasks = [
        Path("/tmp/corpus/models/a/full_graph_000.py"),
        Path("/tmp/corpus/models/b/full_graph_000.py"),
        Path("/tmp/corpus/models/c/full_graph_000.py"),
        Path("/tmp/corpus/models/d/full_graph_000.py"),
    ]
    ledger = _ledger()
    _record_task_duration(
        ledger, task=tasks[0], workload_kind="full_graph", hardware_kind="B200",
        wall_time_s=10.0, status="ok",
    )
    _record_task_duration(
        ledger, task=tasks[2], workload_kind="full_graph", hardware_kind="B200",
        wall_time_s=30.0, status="ok",
    )

    ordered, known = _sort_tasks_by_known_duration(tasks, ledger, "full_graph", "B200")
    assert known == 2
    assert ordered == [tasks[2], tasks[0], tasks[1], tasks[3]]

    ordered, known = _sort_tasks_by_known_duration(tasks, ledger, "full_graph", "GB300")
    assert known == 0
    assert ordered == tasks


def test_record_and_reload_task_duration_ledger(tmp_path):
    ledger_path = tmp_path / "durations.json"
    ledger = _load_task_duration_ledger(ledger_path)
    task = Path("/tmp/corpus/models/model/full_graph_001.py")

    _record_task_duration(
        ledger, task=task, workload_kind="full_graph", hardware_kind="B200",
        wall_time_s=12.0, status="ok",
    )
    ledger_path.write_text(json.dumps(ledger))

    loaded = _load_task_duration_ledger(ledger_path)
    assert loaded["tasks"] == {
        "B200:full_graph:models/model/full_graph_001.py": {
            "wall_time_s": 12.0,
            "status": "ok",
        }
    }


def test_latest_successful_run_replaces_a_past_hang():
    ledger = _ledger()
    task = Path("/c/canonical/a/repro.py")
    for wall_time_s in (900.0, 5.0):
        _record_task_duration(
            ledger, task=task, workload_kind="repro", hardware_kind="B200",
            wall_time_s=wall_time_s, status="ok",
        )
    assert next(iter(ledger["tasks"].values()))["wall_time_s"] == 5.0


def test_fast_failure_does_not_demote_a_known_slow_task():
    ledger = _ledger()
    task = Path("/c/canonical/a/repro.py")
    record = dict(task=task, workload_kind="repro", hardware_kind="B200")
    _record_task_duration(ledger, **record, wall_time_s=60.0, status="ok")
    _record_task_duration(ledger, **record, wall_time_s=2.0, status="failed")
    assert next(iter(ledger["tasks"].values())) == {"wall_time_s": 60.0, "status": "ok"}

    # A failure that ran longer (e.g. a timeout) still raises the estimate.
    _record_task_duration(ledger, **record, wall_time_s=600.0, status="failed")
    assert next(iter(ledger["tasks"].values())) == {
        "wall_time_s": 600.0,
        "status": "failed",
    }


def test_load_task_duration_ledger_rejects_unknown_schema(tmp_path):
    ledger_path = tmp_path / "durations.json"
    ledger_path.write_text('{"schema_version": 2, "tasks": {}}')

    with pytest.raises(ValueError, match="schema_version"):
        _load_task_duration_ledger(ledger_path)


def test_task_duration_key_is_the_same_inside_and_outside_the_repo():
    inside = _REPO_ROOT / "repros" / "canonical" / "foo" / "repro.py"
    outside = Path("/scratch/bundle/canonical/foo/repro.py")
    assert _task_duration_key(inside, "repro", "B200") == _task_duration_key(
        outside, "repro", "B200"
    )
    oracle_dir = _REPO_ROOT / "repros" / "canonical" / "foo"
    assert _task_duration_key(f"{oracle_dir}::SHAPE::s1", "oracle", "B200") == (
        "B200:oracle:canonical/foo::SHAPE::s1"
    )


def test_locked_updates_keep_entries_from_other_writers(tmp_path):
    ledger_path = tmp_path / "durations.json"
    first = Path("/c/canonical/a/repro.py")
    second = Path("/c/canonical/b/repro.py")
    record = dict(workload_kind="repro", hardware_kind="B200", status="ok")
    _update_task_duration_ledger(ledger_path, task=first, wall_time_s=3.0, **record)
    # Another sweep sharing the ledger: its update re-reads the file.
    _update_task_duration_ledger(ledger_path, task=second, wall_time_s=4.0, **record)

    tasks = json.loads(ledger_path.read_text())["tasks"]
    assert set(tasks) == {
        _task_duration_key(first, "repro", "B200"),
        _task_duration_key(second, "repro", "B200"),
    }
    assert (tmp_path / "durations.json.lock").exists()
