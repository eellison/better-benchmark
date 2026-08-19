import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.perf_ab_rollup import (
    _occurrence_sidecars,
    cut,
    fusible_kernel_summary,
    kernel_mover_summary,
    load_arm,
    print_report,
    rollup_models,
    segment_by_suite_mode,
)


def write_sweep(path, patterns):
    path.write_text(
        json.dumps(
            {
                f"/repros/kernel_{pattern}/repro.py": {
                    f"Model_{shape}": measurement
                    for shape, measurement in shapes.items()
                }
                for pattern, shapes in patterns.items()
            }
        )
    )


def write_occurrence(
    path, *, suite="hf", mode="train", model, fusible, extern=None, errors=None
):
    path.write_text(
        json.dumps(
            {
                "suite": suite,
                "mode": mode,
                "model": model,
                "fusible": fusible,
                "extern": extern or [],
                "trace_errors": errors or [],
            }
        )
    )


def test_occurrence_manifest_rejects_incomplete_and_stale_runs(tmp_path):
    sidecar = tmp_path / "model.json"
    sidecar.write_text("{}")
    manifest = {
        "schema_version": 1,
        "status": "incomplete",
        "expected_sidecars": {"hf/infer/model": "model.json"},
        "sidecar_digests": {
            "hf/infer/model": hashlib.sha256(b"{}").hexdigest()
        },
    }
    (tmp_path / "_metadata.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not complete"):
        _occurrence_sidecars(tmp_path)

    manifest["status"] = "complete"
    manifest["expected_sidecars"]["hf/infer/stale"] = "stale.json"
    manifest["sidecar_digests"]["hf/infer/stale"] = "unused"
    (tmp_path / "_metadata.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        _occurrence_sidecars(tmp_path)


def test_occurrence_manifest_rejects_modified_sidecar(tmp_path):
    sidecar = tmp_path / "model.json"
    sidecar.write_text(
        json.dumps({"model": "original", "trace_errors": ["stride mismatch → retry"]}),
        encoding="utf-8",
    )
    identity = "hf/infer/model"
    digest = hashlib.sha256(
        json.dumps(
            json.loads(sidecar.read_text(encoding="utf-8")),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "expected_sidecars": {identity: sidecar.name},
        "sidecar_digests": {identity: digest},
    }
    (tmp_path / "_metadata.json").write_text(json.dumps(manifest))

    assert _occurrence_sidecars(tmp_path) == [sidecar]

    sidecar.write_text(json.dumps({"model": "modified"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match manifest digest"):
        _occurrence_sidecars(tmp_path)


def test_explicit_timing_axes_do_not_fall_back_and_auto_is_compatible(tmp_path):
    sweep = tmp_path / "sweep.json"
    write_sweep(
        sweep,
        {
            "abc": {
                "both": {"compiled_us": 10, "coord_descent_us": 7},
                "compiled": {"compiled_us": 20},
            }
        },
    )

    assert load_arm(sweep, "compiled_us") == {
        ("abc", "both"): 10,
        ("abc", "compiled"): 20,
    }
    assert load_arm(sweep, "coord_descent_us") == {("abc", "both"): 7}
    assert load_arm(sweep) == {
        ("abc", "both"): 7,
        ("abc", "compiled"): 20,
    }


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), "bad", True])
def test_invalid_selected_timing_is_rejected(tmp_path, value):
    sweep = tmp_path / "sweep.json"
    write_sweep(sweep, {"abc": {"shape": {"compiled_us": value}}})
    with pytest.raises(ValueError, match="invalid compiled_us"):
        load_arm(sweep, "compiled_us")


def test_duplicate_pattern_shape_is_rejected(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "/one/kernel_abc/repro.py": {"First_shape": {"compiled_us": 1}},
                "/two/kernel_abc/repro.py": {"Second_shape": {"compiled_us": 2}},
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate pattern/shape"):
        load_arm(sweep, "compiled_us")


def test_exact_kernel_summary_and_bidirectional_movers(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(
        base,
        {
            "abc": {
                "fast": {"compiled_us": 20},
                "slow": {"compiled_us": 10},
                "baseonly": {"compiled_us": 4},
            }
        },
    )
    write_sweep(
        head,
        {
            "abc": {
                "fast": {"compiled_us": 10},
                "slow": {"compiled_us": 12.5},
                "headonly": {"compiled_us": 3},
            }
        },
    )

    summary = fusible_kernel_summary(base, head, "compiled_us")
    movers = kernel_mover_summary(base, head, "compiled_us")

    assert summary["n"] == 2
    assert movers["matched_points"] == 2
    assert movers["base_only_points"] == 1
    assert movers["head_only_points"] == 1
    assert movers["top_improvements"][0]["reduction_pct"] == 50
    assert movers["top_regressions"][0]["reduction_pct"] == -25


def test_kernel_coverage_prints_when_mover_rows_are_suppressed(capsys, tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 5}}})
    movers = kernel_mover_summary(base, head, "compiled_us", limit=0)

    print_report(
        base_json=base,
        head_json=head,
        timing="compiled_us",
        fusible=fusible_kernel_summary(base, head, "compiled_us"),
        kernel_movers=movers,
        per_model=None,
        summaries=None,
        model_coverage=None,
        include_genai=False,
        top_models=0,
    )
    output = capsys.readouterr().out
    assert "KERNEL POINT COVERAGE  matched=1" in output
    assert "TOP KERNEL" not in output
    assert "skipped: no occurrence accounting supplied" in output


def test_disjoint_sweeps_report_coverage_without_a_summary(capsys, tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"baseonly": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"headonly": {"compiled_us": 5}}})
    movers = kernel_mover_summary(base, head, "compiled_us", limit=0)

    print_report(
        base_json=base,
        head_json=head,
        timing="compiled_us",
        fusible=None,
        kernel_movers=movers,
        per_model=None,
        summaries=None,
        model_coverage=None,
        include_genai=False,
        top_models=0,
    )

    output = capsys.readouterr().out
    assert "(no matched entries)" in output
    assert "matched=0  base-only=1  head-only=1" in output


def test_nonfinite_derived_metric_is_rejected(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"shape": {"compiled_us": 1e-300}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 1e300}}})

    with pytest.raises(ValueError, match="reduction is not finite"):
        fusible_kernel_summary(base, head, "compiled_us")


def test_nonfinite_geomean_percentage_is_rejected(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"shape": {"compiled_us": 1e307}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 1}}})

    with pytest.raises(ValueError, match="speedup percent is not finite"):
        fusible_kernel_summary(base, head, "compiled_us")


def test_incomplete_models_are_excluded_not_partially_priced(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    occdir = tmp_path / "occurrences"
    occdir.mkdir()
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 5}}})
    write_occurrence(
        occdir / "complete.json",
        model="complete",
        fusible={"abc": {"shape": 1}},
    )
    write_occurrence(
        occdir / "fallback.json",
        model="fallback",
        fusible={"abc": {"other": 2}},
    )
    write_occurrence(
        occdir / "unmatched.json",
        model="unmatched",
        fusible={"unknown": {"shape": 3}},
    )
    write_occurrence(
        occdir / "extern.json",
        model="extern",
        fusible={"abc": {"shape": 1}},
        extern=[{"target": "mm", "count": 4, "baseline_us": None}],
    )
    write_occurrence(
        occdir / "trace.json",
        model="trace",
        fusible={"abc": {"shape": 1}},
        errors=["graph failed"],
    )

    models, coverage = rollup_models(base, head, occdir, timing="compiled_us")

    assert models["hf/train/complete"] == 50
    assert all(
        models[f"hf/train/{name}"] is None
        for name in ("fallback", "unmatched", "extern", "trace")
    )
    assert coverage["models"] == {"total": 5, "included": 1, "excluded": 4}
    assert coverage["kernel_occurrences"] == {
        "exact": 3,
        "paired_pattern_fallback": 2,
        "unmatched": 3,
    }
    assert coverage["extern_occurrences"]["unpriced"] == 4
    assert coverage["per_model"]["hf/train/extern"]["exclusion_reasons"] == [
        "unpriced_extern"
    ]
    assert coverage["excluded_entries"] == [
        "hf/train/extern",
        "hf/train/fallback",
        "hf/train/trace",
        "hf/train/unmatched",
    ]


def test_model_without_any_priced_work_is_excluded_with_reason(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    occdir = tmp_path / "occurrences"
    occdir.mkdir()
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 5}}})
    write_occurrence(occdir / "empty.json", model="empty", fusible={})

    models, coverage = rollup_models(base, head, occdir, timing="compiled_us")

    assert models["hf/train/empty"] is None
    assert coverage["per_model"]["hf/train/empty"]["exclusion_reasons"] == [
        "no_priced_baseline"
    ]


def test_occurrence_weight_overflow_is_rejected(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    occdir = tmp_path / "occurrences"
    occdir.mkdir()
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 5}}})
    write_occurrence(
        occdir / "huge.json",
        model="huge",
        fusible={"abc": {"shape": 10**400}},
    )

    with pytest.raises(ValueError, match="model baseline is not finite"):
        rollup_models(base, head, occdir, timing="compiled_us")


def test_genai_is_microbenchmark_suite_not_model_substring():
    models = {
        "hf/train/my_genai_model": 1,
        "genai/static/softmax": 10,
        "GenAI/static/cross_entropy": None,
    }
    assert cut(models, genai_excl=True)["n"] == 1
    rows = dict(segment_by_suite_mode(models))
    assert rows["genai/static"]["n"] == 1
    assert rows["genai/static"]["excluded"] == 1


def test_bottom_models_are_ordered_worst_first(capsys, tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 10}}})
    per_model = {
        "hf/train/winner": 2,
        "hf/train/mild": -1,
        "hf/train/worst": -3,
    }
    print_report(
        base_json=base,
        head_json=head,
        timing="compiled_us",
        fusible=fusible_kernel_summary(base, head, "compiled_us"),
        kernel_movers=kernel_mover_summary(base, head, "compiled_us"),
        per_model=per_model,
        summaries={
            "genai_excl": cut(per_model),
            "genai_incl": cut(per_model, genai_excl=False),
        },
        model_coverage={
            "models": {"included": 3, "total": 3},
            "microbenchmarks": {"included": 0, "total": 0},
            "kernel_occurrences": {
                "exact": 3,
                "paired_pattern_fallback": 0,
                "unmatched": 0,
            },
            "extern_occurrences": {"priced": 0, "unpriced": 0},
            "excluded_entries": [],
            "per_model": {},
        },
        include_genai=False,
        top_models=2,
    )
    bottom = capsys.readouterr().out.split("BOTTOM MODELS", 1)[1]
    assert bottom.index("hf/train/worst") < bottom.index("hf/train/mild")


def test_cli_runs_without_occurrence_artifacts(tmp_path):
    base = tmp_path / "base.json"
    head = tmp_path / "head.json"
    write_sweep(base, {"abc": {"shape": {"compiled_us": 10}}})
    write_sweep(head, {"abc": {"shape": {"compiled_us": 5}}})
    script = Path(__file__).parents[1] / "scripts/perf_ab_rollup.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--base",
            str(base),
            "--head",
            str(head),
            "--timing",
            "compiled_us",
            "--top-kernels",
            "0",
            "--json",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["fusible_kernel"]["n"] == 1
    assert report["kernel_movers"]["matched_points"] == 1
    assert report["coverage"] is None
    assert report["per_model"] == {}
    assert report["include_genai"] is False
    assert report["include_genai_in_rankings"] is False
