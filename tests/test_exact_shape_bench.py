import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_parallel import (  # noqa: E402
    _SHAPE_TASK_SEP,
    _build_run_summary,
    _canonical_shape_hash_labels,
    _filter_exact_shape_tasks,
    _merge_repro_scopes,
    _remove_recovered_shape_failures,
    _repro_status_counts,
    _regroup_sharded_repro_failures,
    _regroup_sharded_repro_results,
    _shape_items_for_task,
    _write_results_output,
    load_benchmark_set_selection,
)


def _write_repro(
    canonical_dir: Path,
    name: str,
    points: list[tuple[str, str]] | None,
) -> Path:
    repro_dir = canonical_dir / name
    repro_dir.mkdir(parents=True)
    repro = repro_dir / "repro.py"
    repro.write_text("# test repro\n")
    if points is not None:
        (repro_dir / "shapes.json").write_text(
            json.dumps(
                {
                    "points": [
                        {
                            "shape_hash": shape_hash,
                            "models": {f"hf/train/{model}": {"occurrences": 1}},
                            "inputs": [],
                        }
                        for shape_hash, model in points
                    ]
                }
            )
        )
    return repro


def _write_manifest(path: Path, entries: list) -> Path:
    path.write_text(json.dumps({"benchmarks": entries}))
    return path


def test_manifest_rejects_multiple_selector_fields(tmp_path):
    canonical = tmp_path / "canonical"
    _write_repro(canonical, "family", [("hash_a", "ModelA")])
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {
                "repro": "family",
                "shape": "ModelA_hash_a",
                "shape_labels": ["ModelA_hash_a"],
            }
        ],
    )

    with pytest.raises(ValueError, match="multiple shape selectors"):
        load_benchmark_set_selection(manifest, canonical_dir=canonical)


def test_shape_and_shape_labels_select_only_requested_expanded_points(tmp_path):
    canonical = tmp_path / "canonical"
    repro = _write_repro(
        canonical,
        "family",
        [
            ("hash_a", "ModelA"),
            ("hash_b", "ModelB"),
            ("hash_c", "ModelC"),
        ],
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            {"repro": "family", "shape": "ModelA_hash_a"},
            {
                "repro": "family",
                "shape_labels": ["ModelC_hash_c"],
            },
        ],
    )

    selection = load_benchmark_set_selection(manifest, canonical_dir=canonical)

    assert selection.repros == (repro,)
    assert selection.n_entries == 2
    assert selection.n_points == 2
    assert selection.partial_repros == (str(repro),)
    assert selection.tasks == (
        f"{repro}{_SHAPE_TASK_SEP}ModelA_hash_a",
        f"{repro}{_SHAPE_TASK_SEP}ModelC_hash_c",
    )


def test_default_selector_normalizes_but_preserves_worker_task_token(tmp_path):
    canonical = tmp_path / "canonical"
    repro = _write_repro(canonical, "default_only", None)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"repro": "default_only", "shape": "__default__"}],
    )

    selection = load_benchmark_set_selection(manifest, canonical_dir=canonical)

    assert selection.tasks == (f"{repro}{_SHAPE_TASK_SEP}__default__",)
    assert selection.partial_repros == (str(repro),)

    regrouped = _regroup_sharded_repro_results(
        {selection.tasks[0]: {"__default__": {"compiled_us": 1.0}}}
    )
    assert regrouped == {str(repro): {"default": {"compiled_us": 1.0}}}

    failures = _regroup_sharded_repro_failures(
        regrouped,
        {selection.tasks[0]: {"status": "failed"}},
    )
    assert failures == {f"{repro}{_SHAPE_TASK_SEP}default": {"status": "failed"}}


def test_unqualified_manifest_entry_is_a_complete_family_run(tmp_path):
    canonical = tmp_path / "canonical"
    repro = _write_repro(
        canonical,
        "family",
        [("hash_a", "ModelA"), ("hash_b", "ModelB")],
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"repro": "family", "shape": None}],
    )

    selection = load_benchmark_set_selection(manifest, canonical_dir=canonical)

    assert selection.partial_repros == ()
    assert selection.tasks == (
        f"{repro}{_SHAPE_TASK_SEP}ModelA_hash_a",
        f"{repro}{_SHAPE_TASK_SEP}ModelB_hash_b",
    )


def test_missing_shape_label_fails_instead_of_broadening(tmp_path):
    canonical = tmp_path / "canonical"
    _write_repro(canonical, "family", [("hash_a", "ModelA")])
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"repro": "family", "shape_labels": ["missing_label"]}],
    )

    with pytest.raises(ValueError, match="did not resolve exactly"):
        load_benchmark_set_selection(manifest, canonical_dir=canonical)


def test_shape_hashes_resolve_exactly_through_shapes_json(tmp_path):
    canonical = tmp_path / "canonical"
    repro = _write_repro(
        canonical,
        "family",
        [
            ("abc", "Short"),
            ("prefix_abc", "Long"),
        ],
    )
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"repro": "family", "shape_hashes": ["abc"]}],
    )

    assert _canonical_shape_hash_labels(repro) == {
        "abc": "Short_abc",
        "prefix_abc": "Long_prefix_abc",
    }
    selection = load_benchmark_set_selection(manifest, canonical_dir=canonical)
    assert selection.tasks == (f"{repro}{_SHAPE_TASK_SEP}Short_abc",)

    missing = _write_manifest(
        tmp_path / "missing.json",
        [{"repro": "family", "shape_hashes": ["bc"]}],
    )
    with pytest.raises(ValueError, match="does not contain requested canonical"):
        load_benchmark_set_selection(missing, canonical_dir=canonical)


def test_filter_exact_shape_tasks_reports_every_missing_repro():
    tasks = [
        f"/a/repro.py{_SHAPE_TASK_SEP}shape_a",
        f"/b/repro.py{_SHAPE_TASK_SEP}shape_b",
    ]

    with pytest.raises(ValueError) as exc:
        _filter_exact_shape_tasks(
            tasks,
            {
                "/a/repro.py": {"missing_a"},
                "/b/repro.py": {"missing_b"},
            },
        )

    assert "/a/repro.py: missing_a" in str(exc.value)
    assert "/b/repro.py: missing_b" in str(exc.value)


def test_persistent_worker_shape_selection_is_exact_and_fail_closed():
    configs = {"shape_a": {"inputs": ["a"]}, "shape_b": {"inputs": ["b"]}}

    assert _shape_items_for_task(configs, "shape_b", True) == [
        ("shape_b", configs["shape_b"])
    ]
    assert _shape_items_for_task(configs, None, True) == list(configs.items())
    assert _shape_items_for_task(configs, None, False) == [(None, None)]
    assert _shape_items_for_task({}, "__default__", True) == [(None, None)]
    with pytest.raises(ValueError, match="unavailable"):
        _shape_items_for_task(configs, "missing", True)
    with pytest.raises(ValueError, match="named shape configs exist"):
        _shape_items_for_task(configs, "__default__", True)


def test_incomplete_complete_family_degrades_to_partial_merge():
    repro = "/canonical/family/repro.py"
    task_a = f"{repro}{_SHAPE_TASK_SEP}shape_a"
    task_b = f"{repro}{_SHAPE_TASK_SEP}shape_b"

    partial, complete, incomplete = _merge_repro_scopes(
        {task_a, task_b},
        {task_a},
        set(),
    )

    assert partial == {repro}
    assert complete == set()
    assert incomplete == {repro}


def test_summary_retains_point_counts_and_exposes_repro_counts():
    summary = _build_run_summary(
        point_total=5,
        point_ok=3,
        point_failed=2,
        point_skipped=0,
        elapsed=12.5,
        repro_total=2,
        repro_ok=1,
        repro_failed=1,
        repro_skipped=0,
    )

    assert summary["total"] == 2
    assert summary["ok"] == 1
    assert summary["points"] == {
        "total": 5,
        "ok": 3,
        "failed": 2,
        "skipped": 0,
    }
    assert summary["repros"] == {
        "total": 2,
        "ok": 1,
        "failed": 1,
        "skipped": 0,
    }


def test_repro_status_counts_give_failures_precedence_over_skips():
    repro = "/canonical/family/repro.py"
    counts = _repro_status_counts(
        {},
        {
            f"{repro}{_SHAPE_TASK_SEP}shape_a": {"status": "skipped"},
            f"{repro}{_SHAPE_TASK_SEP}shape_b": {"status": "failed"},
        },
        total=1,
    )

    assert counts == (1, 0, 1, 0)


def test_output_writer_keeps_point_and_repro_accounting_separate(tmp_path):
    repro = "/canonical/family/repro.py"
    output = tmp_path / "results.json"
    _write_results_output(
        output,
        {
            repro: {
                "shape_a": {"compiled_us": 1.0},
                "shape_b": {"compiled_us": 2.0},
            }
        },
        {f"{repro}{_SHAPE_TASK_SEP}shape_c": {"status": "failed"}},
        total=3,
        done=2,
        failed=1,
        skipped=0,
        elapsed=3.5,
        workload_kind="repro",
        repro_total=1,
        repro_done=1,
        repro_failed=0,
        repro_skipped=0,
    )

    payload = json.loads(output.read_text())
    assert payload["__summary__"]["points"]["total"] == 3
    assert payload["__summary__"]["repros"]["total"] == 1
    assert payload["_metadata"]["n_results"] == 1
    assert payload["_metadata"]["n_repros"] == 1


def test_all_selected_shape_failures_remain_point_qualified():
    repro = "/canonical/family/repro.py"
    failures = _regroup_sharded_repro_failures(
        {},
        {
            f"{repro}{_SHAPE_TASK_SEP}shape_a": {"status": "failed"},
            f"{repro}{_SHAPE_TASK_SEP}shape_b": {"status": "failed"},
        },
    )

    assert set(failures) == {
        f"{repro}{_SHAPE_TASK_SEP}shape_a",
        f"{repro}{_SHAPE_TASK_SEP}shape_b",
    }


def test_recovering_one_shape_preserves_failed_siblings():
    repro = "/canonical/family/repro.py"
    failures = {
        repro: {
            "status": "failed",
            "error": "worker failed",
            "failed_shapes": ["shape_a", "shape_b", "__default__"],
        }
    }

    _remove_recovered_shape_failures(
        failures,
        repro,
        {"shape_a", "default"},
    )

    assert set(failures) == {f"{repro}{_SHAPE_TASK_SEP}shape_b"}
    assert failures[f"{repro}{_SHAPE_TASK_SEP}shape_b"]["failed_shape"] == "shape_b"
    assert "failed_shapes" not in failures[f"{repro}{_SHAPE_TASK_SEP}shape_b"]
