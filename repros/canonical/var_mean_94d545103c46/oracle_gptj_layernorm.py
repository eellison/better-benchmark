"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GPT-J residual LayerNorm scope in one shape-specialized Triton row kernel, including the two `[128,4096]` views, three fp32 activation adds, fp32 `var_mean(correction=0, keepdim=True)` over hidden size 4096, affine scale/bias epilogue, final `[128,4096]` view, and sibling `rsqrt(var + 1e-5) / 4096` output, whereas tuned Inductor already lowers this fixed normalization region into the same mandatory memory-traffic envelope with a fused norm template; Inductor cannot materially improve it through local scheduler fusion because the remaining work is dominated by required activation/affine reads, one wide row reduction, rsqrt, output stores, and launch overhead rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope norm case unless broader normalization-template or launch-overhead work moves the baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
HIDDEN = 4096
EPS = 1.0e-5
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _gptj_residual_layernorm_kernel(
        mm_ptr,
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        x0 = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x0 + x1 + x2

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)
        tl.store(invstd_div_ptr + row, invstd / hidden)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    mm_111, addmm_55, add_216, weight, bias, shape0, shape1, shape2 = inputs
    tensors = (mm_111, addmm_55, add_216, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (ROWS, HIDDEN),
        (ROWS, HIDDEN),
        (1, ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensors, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if tuple(int(dim) for dim in shape0) != (1, ROWS, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if tuple(int(dim) for dim in shape1) != (1, ROWS, HIDDEN):
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")
    if tuple(int(dim) for dim in shape2) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape parameter: {shape2!r}")

    return mm_111, addmm_55, add_216, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([4096], f32), T([4096], f32), S([1, 128, 4096]), S([1, 128, 4096]), S([128, 4096]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward residual LayerNorm computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gptj_layernorm.py")

    mm_111, addmm_55, add_216, weight, bias = _validate_inputs(inputs)
    out = torch.empty((ROWS, HIDDEN), device=mm_111.device, dtype=torch.float32)
    invstd_div = torch.empty((1, ROWS, 1), device=mm_111.device, dtype=torch.float32)
    _gptj_residual_layernorm_kernel[(ROWS,)](
        mm_111,
        addmm_55,
        add_216,
        weight,
        bias,
        out,
        invstd_div,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=HIDDEN,
        num_warps=8,
    )
    return out, invstd_div


def main() -> None:
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
