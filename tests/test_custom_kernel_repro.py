import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_parallel import _persistent_worker_script  # noqa: E402
from repro_harness import bench_custom_kernel_point  # noqa: E402


def test_bench_parallel_worker_dispatches_custom_kernels():
    args = {"root": str(ROOT), "all_shapes": False, "no_cd": True, "n_warmup": 1,
            "n_rep": 1, "strict_gpu_lock": False, "workload_kind": "repro"}
    script = _persistent_worker_script("0", args)
    compile(script, "worker", "exec")
    assert "if custom_kernel_meta(repro_path) is not None:" in script
    assert script.count("bench_custom_kernel_point(") == 1


class _Repro(torch.nn.Module):
    """Stands in for one custom-kernel launch: writes its output in place."""

    def prepare(self, inputs):
        self.prepared = True

    def forward(self, offsets, x, out):
        torch.mul(x, 2.0, out=out)
        return (out,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bench_custom_kernel_point_reports_kernel_times_not_compiled():
    from triton.testing import do_bench

    cfg = {"inputs": [
        {"kind": "tensor", "name": "offsets", "shape": [9], "dtype": "int64",
         "gen": {"kind": "jagged_offsets", "total": {"value": 4096}, "max_len": 1024,
                 "distribution": "uniform"}},
        {"kind": "tensor", "name": "x", "shape": [4096, 64], "dtype": "float32"},
        {"kind": "tensor", "name": "out", "shape": [4096, 64], "dtype": "float32"},
    ]}
    result = bench_custom_kernel_point(
        _Repro, cfg, bench=lambda fn: do_bench(fn, warmup=1, rep=5, return_mode="min"), rounds=2)
    assert result["workload"] == "custom_kernel"
    assert result["kernel_us"] > 0 and result["kernel_device_us"] > 0
    assert "compiled_us" not in result and "dev_run" not in result
