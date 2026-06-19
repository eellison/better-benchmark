"""GPU acceptance test for --dynamic / --bind benchmarking (standalone).

Reconstructs the dynamic_shapes_capture_design.md §1.4 result through the
NEW harness machinery: a GroupNorm-like repro (opacus var_mean_b1feb9d09685
shape family) with a fabricated dynamic shapes.json (s16/s82, hint 16,
range [2, inf)) is measured three ways:

  (a) static compile at the 16x16 point        (1 persistent kernel)
  (b) dynamic compile at the SAME 16x16 point  (looped reduction, 2 kernels)
  (c) the SAME dynamic artifact at 24x24 via --bind, no recompilation

Asserts: (a) < (b) at the same point (the design doc saw 13.7us vs
35.5us), and (c) reused the compiled artifact (recompile detected via
torch._dynamo unique_graphs counter deltas inside the harness).

GPU-required. Usage:
    CUDA_VISIBLE_DEVICES=1 INDUCTOR_GPU_BENCH_LOCK=1 \
        python scripts/test_dynamic_bench.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from repro_harness import benchmark_repro, make_inputs_from_config, \
    load_shape_configs  # noqa: E402


class Repro(torch.nn.Module):
    """GroupNorm-affine pattern of var_mean_b1feb9d09685 (opacus_cifar10).

    view -> var_mean -> normalize -> view back -> scale + shift. With
    static shapes Inductor emits ONE persistent reduction; with symbolic
    rnumel (2*s16*s82) it falls back to a looped Welford reduction plus a
    separate pointwise epilogue — the 2.6x gap the design doc measured.
    """

    def forward(self, arg2_1, arg0_1, arg1_1, _shape_param_0):
        view = torch.ops.aten.view.default(arg2_1, _shape_param_0)
        var_mean = torch.ops.aten.var_mean.correction(
            view, [2, 3], correction=0, keepdim=True)
        var, mean = var_mean[0], var_mean[1]
        sub = torch.ops.aten.sub.Tensor(view, mean)
        rsqrt = torch.ops.aten.rsqrt.default(
            torch.ops.aten.add.Tensor(var, 1e-05))
        normed = torch.ops.aten.mul.Tensor(sub, rsqrt)
        back = torch.ops.aten.view.default(normed, arg2_1.shape)
        w = torch.ops.aten.unsqueeze.default(
            torch.ops.aten.unsqueeze.default(
                torch.ops.aten.unsqueeze.default(arg0_1, 0), 2), 3)
        b = torch.ops.aten.unsqueeze.default(
            torch.ops.aten.unsqueeze.default(
                torch.ops.aten.unsqueeze.default(arg1_1, 0), 2), 3)
        return torch.ops.aten.add.Tensor(
            torch.ops.aten.mul.Tensor(back, w), b)


SHAPES_JSON = {
    "symbols": {
        "s16": {"hint": 16, "range": [2, None]},
        "s82": {"hint": 16, "range": [2, None]},
    },
    "points": [{
        "shape_hash": "dynexp01",
        "bindings": {"s16": 16, "s82": 16},
        "models": {"torchbench/infer/opacus_like": {"occurrences": 1}},
        "inputs": [
            [[64, 64, "s16", "s82"], "f32"],
            [[64], "f32"],
            [[64], "f32"],
            ["S", [64, 32, 2, "s16*s82"]],
        ],
    }],
}


def make_inputs(shape_config=None):
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    # default = the captured hint point
    repro_file = os.environ["_DYN_BENCH_REPRO_FILE"]
    configs = load_shape_configs(repro_file)
    return make_inputs_from_config(next(iter(configs.values())))


def _single_row(results: dict, binding_str: str, mode: str) -> dict:
    rows = [r for r in results.values()
            if r["mode"] == mode
            and ",".join(f"{k}={v}" for k, v in
                         sorted((r["binding"] or {}).items())) == binding_str]
    assert len(rows) == 1, (binding_str, mode, list(results))
    return rows[0]


def main() -> int:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    common = ["--gpu", gpu]

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "shapes.json").write_text(json.dumps(SHAPES_JSON))
        repro_file = str(d / "repro.py")
        Path(repro_file).write_text("# fixture stub: see test_dynamic_bench")
        os.environ["_DYN_BENCH_REPRO_FILE"] = repro_file

        # (a) static compile at the 16x16 point
        static = benchmark_repro(
            repro_file, Repro, make_inputs,
            args=["--bind", "s16=16,s82=16", *common])
        row_a = _single_row(static, "s16=16,s82=16", "static")

        # (b)+(c): ONE dynamic artifact measured at both family points
        dynamic = benchmark_repro(
            repro_file, Repro, make_inputs,
            args=["--dynamic",
                  "--bind", "s16=16,s82=16",
                  "--bind", "s16=24,s82=24", *common])
        row_b = _single_row(dynamic, "s16=16,s82=16", "dynamic")
        row_c = _single_row(dynamic, "s16=24,s82=24", "dynamic")

    print("\n=== dynamic-bench acceptance ===")
    print(f"(a) static  16x16: {row_a['compiled_us']:8.1f} us  "
          f"kernels={row_a['n_kernels']}")
    print(f"(b) dynamic 16x16: {row_b['compiled_us']:8.1f} us  "
          f"kernels={row_b['n_kernels']}")
    print(f"(c) dynamic 24x24: {row_c['compiled_us']:8.1f} us  "
          f"recompiled={row_c['recompiled']}")

    failures = []
    # The point of the work: the dynamic artifact is a measurably
    # different (slower) kernel at the SAME family point.
    if not row_a["compiled_us"] < row_b["compiled_us"]:
        failures.append(
            f"static ({row_a['compiled_us']:.1f}us) not faster than "
            f"dynamic ({row_b['compiled_us']:.1f}us) at 16x16")
    # (c) must reuse (b)'s artifact — no recompilation between points.
    if row_c["recompiled"] is not False:
        failures.append(
            f"dynamic 24x24 point recompiled (recompiled="
            f"{row_c['recompiled']}) — artifact not reused")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
