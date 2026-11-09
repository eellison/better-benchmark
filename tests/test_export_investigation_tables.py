import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_investigation_tables.py"
SPEC = importlib.util.spec_from_file_location("export_investigation_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


_MISSING = object()


def _result(repro: Path, *, n_kernels=_MISSING, compiled_us: float = 20.0, **extra):
    record = {
        "compiled_us": compiled_us,
        "coord_descent_us": None,
        "memcopy_sol_us": 10.0,
        "gap_default": compiled_us / 10.0,
        "gap_cd": None,
        "total_bytes": 4096,
        "sol_mitigation_factors": ["indirect_read"],
        **extra,
    }
    if n_kernels is not _MISSING:
        record["n_kernels"] = n_kernels
    return {str(repro): {"default": record}}


def _repro(tmp_path: Path, name: str, n_ops: int) -> Path:
    directory = tmp_path / "canonical" / name
    directory.mkdir(parents=True)
    repro = directory / "repro.py"
    repro.write_text("# repro\n")
    (directory / "meta.json").write_text(
        json.dumps({"n_ops": n_ops, "origin_ops": ["aten.add.Tensor"]})
    )
    return repro


def test_sol_comparison_classes_separate_multi_kernel_mega_regions(tmp_path):
    single = _repro(tmp_path, "pointwise_000000000001", 8)
    multi = _repro(tmp_path, "sum_000000000002", 40)
    mega = _repro(tmp_path, "sum_sum_000000000003", 300)
    eliminated = _repro(tmp_path, "pointwise_000000000004", 4)

    payload = {}
    payload.update(_result(single, n_kernels=1))
    payload.update(_result(multi, n_kernels=4))
    payload.update(_result(mega, n_kernels=20, compiled_us=100.0))
    payload.update(_result(eliminated, n_kernels=0))

    rows = MODULE.baseline_rows(payload)
    by_id = {row["repro_id"]: row for row in rows}

    assert by_id[single.parent.name]["sol_comparison_class"] == (
        "single_kernel_copy_roofline"
    )
    assert by_id[multi.parent.name]["sol_comparison_class"] == (
        "multi_kernel_logical_io"
    )
    assert by_id[mega.parent.name]["sol_comparison_class"] == (
        "multi_kernel_mega_region"
    )
    assert by_id[mega.parent.name]["combo_sweep_recommended"] is True
    assert by_id[eliminated.parent.name]["sol_comparison_class"] == (
        "no_generated_kernel"
    )
    assert by_id[eliminated.parent.name]["n_kernels"] == 0

    raw, single_kernel, multi_kernel = MODULE.split_sol_gap_candidates(rows)
    assert len(raw) == 4
    assert [row["repro_id"] for row in single_kernel] == [single.parent.name]
    assert [row["repro_id"] for row in multi_kernel] == [
        mega.parent.name,
        multi.parent.name,
    ]
    assert multi_kernel[0]["multi_kernel_priority_rank"] == 1
    assert "multi_kernel_priority_rank" not in by_id[mega.parent.name]


def test_mega_region_requires_both_large_graph_and_launch_forest():
    assert MODULE.classify_sol_comparison(20, 300) == "multi_kernel_mega_region"
    assert MODULE.classify_sol_comparison(20, 40) == "multi_kernel_logical_io"
    assert MODULE.classify_sol_comparison(4, 300) == "multi_kernel_logical_io"


@pytest.mark.parametrize("n_kernels", [_MISSING, None, "7", 7.0, True, -1])
def test_unknown_kernel_count_is_not_a_single_kernel(tmp_path, n_kernels):
    repro = _repro(tmp_path, "sum_000000000009", 300)
    rows = MODULE.baseline_rows(_result(repro, n_kernels=n_kernels, compiled_us=50.0))

    assert rows[0]["n_kernels"] is None
    assert rows[0]["sol_comparison_class"] == "unknown_kernel_count"
    assert rows[0]["single_copy_roofline_applicable"] is False
    assert math.isnan(rows[0]["launch_adjusted_gap_3p0"])
    _, single_kernel, multi_kernel = MODULE.split_sol_gap_candidates(rows)
    assert single_kernel == [] and multi_kernel == []
    assert MODULE.launch_adjusted_priority_rows(rows) == []


def _bench_parallel_output(tmp_path, results, inductor_config):
    """Results written the way bench_parallel --output writes them."""
    import sys
    from unittest import mock

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import bench_parallel

    output = tmp_path / "sweep.json"
    with mock.patch.object(bench_parallel, "_git_query", return_value="deadbeef"), \
            mock.patch.object(bench_parallel, "_pytorch_provenance", return_value=("abc", None)):
        bench_parallel._write_results_output(
            output, results, {}, total=len(results), done=len(results), failed=0,
            elapsed=1.0, workload_kind="repro", extra_inductor_config=inductor_config,
            config_metadata={"coordinate_descent": True},
        )
    return json.loads(output.read_text())


@pytest.mark.parametrize("multi_kernel", [1, 2, 3])
def test_multi_kernel_variants_are_not_counted_as_launches(tmp_path, multi_kernel):
    single = _repro(tmp_path, "sum_000000000010", 8)
    variants = _repro(tmp_path, "sum_000000000011", 8)
    results = {**_result(single, n_kernels=1), **_result(variants, n_kernels=3)}
    payload = _bench_parallel_output(
        tmp_path, results, {"triton.multi_kernel": multi_kernel}
    )
    by_id = {row["repro_id"]: row for row in MODULE.baseline_rows(payload)}

    assert by_id[single.parent.name]["sol_comparison_class"] == "single_kernel_copy_roofline"
    assert by_id[variants.parent.name]["sol_comparison_class"] == "unknown_launch_count"
    _, _, multi_kernel_rows = MODULE.split_sol_gap_candidates(list(by_id.values()))
    assert multi_kernel_rows == []


def test_without_multi_kernel_generated_count_is_the_launch_count(tmp_path):
    variants = _repro(tmp_path, "sum_000000000011", 8)
    payload = _bench_parallel_output(tmp_path, _result(variants, n_kernels=3), None)
    (row,) = MODULE.baseline_rows(payload)
    assert row["sol_comparison_class"] == "multi_kernel_logical_io"
