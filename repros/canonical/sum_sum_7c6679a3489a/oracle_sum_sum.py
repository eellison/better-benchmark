"""
Full-scope oracle for sum_sum_7c6679a3489a.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle consumes the
same five tensor inputs and four shape parameters as repro.py, computes both
per-channel reductions over the logical `[1, 256, 128]` view, and immediately
uses those finalized reductions to write the returned `float32[256, 1, 1, 1]`
side tensor and full `float32[256, 128, 1, 1]` tensor in one Triton program per
channel. Inductor cannot do this today because its scheduler does not form one
full-scope multi-output reduction template that shares sibling reductions and
sinks the finalized scalars into the dependent full-output epilogue across the
reshape-only views; the fix is SCHEDULER_FUSION: add a scheduler/codegen
template for compatible sibling reductions with a dependent materializing
epilogue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

C = 256
R = 128
INV_R = 0.0078125
WEIGHT_SCALE = 0.08838834764831845
OUT0_SHAPE = (C, 1, 1, 1)
OUT0_STRIDE = (1, 1, 1, 1)
OUT1_SHAPE = (C, R, 1, 1)
OUT1_STRIDE = (R, 1, 1, 1)
BLOCK_R = 128
NUM_WARPS = 4
CLASSIFICATION = "SCHEDULER_FUSION"


from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _sum_sum_epilogue_kernel(
        getitem_ptr,
        primals_14_ptr,
        mean_ptr,
        squeeze_ptr,
        primals_15_ptr,
        out0_ptr,
        out1_ptr,
        BLOCK: tl.constexpr,
        INV_R_: tl.constexpr,
        WEIGHT_SCALE_: tl.constexpr,
    ):
        c = tl.program_id(0)
        r = tl.arange(0, BLOCK)
        offsets = c * BLOCK + r

        a = tl.load(getitem_ptr + offsets).to(tl.float32)
        b = tl.load(primals_14_ptr + offsets).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        squeeze = tl.load(squeeze_ptr + c).to(tl.float32)
        gain = tl.load(primals_15_ptr + c).to(tl.float32)

        centered = b - mean
        sum_a = tl.sum(a, axis=0)
        sum_ax = tl.sum(a * centered, axis=0)

        avg_a = sum_a * INV_R_
        correction_scale = sum_ax * INV_R_ * squeeze * squeeze
        output_scale = squeeze * gain * WEIGHT_SCALE_

        full = (a - centered * correction_scale - avg_a) * output_scale
        reduced = sum_ax * squeeze * WEIGHT_SCALE_

        tl.store(out1_ptr + offsets, full)
        tl.store(out0_ptr + c, reduced)


def _require_tensor(
    tensor: object,
    name: str,
    shape: tuple[int, ...],
    *,
    contiguous: bool = True,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} expected shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} expected dtype torch.float32, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for this oracle")
    return tensor


@oracle_impl(hardware="H100", shapes="(T([256, 128, 1, 1], f32), T([256, 128, 1, 1], f32), T([1, 256, 1], f32), T([256], f32), T([256, 1, 1, 1], f32), S([1, 256, 128]), S([1, 256, -1]), S([256, 1, 1, 1]), S([256, 128, 1, 1]))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with one fused reduction/epilogue kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum_sum.py")
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    (
        getitem_343,
        primals_14,
        unsqueeze_474,
        squeeze_9,
        primals_15,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs

    getitem_343 = _require_tensor(getitem_343, "getitem_343", OUT1_SHAPE)
    primals_14 = _require_tensor(primals_14, "primals_14", OUT1_SHAPE)
    unsqueeze_474 = _require_tensor(unsqueeze_474, "unsqueeze_474", (1, C, 1))
    squeeze_9 = _require_tensor(squeeze_9, "squeeze_9", (C,))
    primals_15 = _require_tensor(primals_15, "primals_15", OUT0_SHAPE)

    out0 = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=getitem_343.device,
        dtype=torch.float32,
    )
    out1 = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=getitem_343.device,
        dtype=torch.float32,
    )

    _sum_sum_epilogue_kernel[(C,)](
        getitem_343,
        primals_14,
        unsqueeze_474,
        squeeze_9,
        primals_15,
        out0,
        out1,
        BLOCK=BLOCK_R,
        INV_R_=INV_R,
        WEIGHT_SCALE_=WEIGHT_SCALE,
        num_warps=NUM_WARPS,
    )
    return out0, out1


def _check_layouts(outputs: object, expected_outputs: object) -> bool:
    if not isinstance(outputs, (tuple, list)) or not isinstance(expected_outputs, (tuple, list)):
        print("  layout: FAIL (expected tuple/list outputs)")
        return False
    if len(outputs) != len(expected_outputs):
        print(f"  layout: FAIL (oracle outputs={len(outputs)} eager outputs={len(expected_outputs)})")
        return False

    ok = True
    for idx, (got, expected) in enumerate(zip(outputs, expected_outputs)):
        layout_ok = (
            isinstance(got, torch.Tensor)
            and isinstance(expected, torch.Tensor)
            and tuple(got.shape) == tuple(expected.shape)
            and got.stride() == expected.stride()
            and got.dtype == expected.dtype
        )
        print(
            f"  output {idx} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(got.shape)} stride={got.stride()} dtype={got.dtype})"
        )
        ok = ok and layout_ok
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        with torch.no_grad():
            expected = instance(*inputs)
            actual = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_layouts(actual, expected)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
