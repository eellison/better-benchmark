"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Repro.forward fp32 [128, 2] log_softmax region in one shape-specialized Triton kernel, covering the row amax, subtract, exp, sum, log, and final subtract with no omitted surrounding attention or loss ops because the captured repro contains none, whereas Inductor currently lowers the same full scope through its generic small reduction/pointwise code path and can match or beat this hand kernel at launch-scale timings; Inductor cannot do this today because its pattern/codegen path does not recognize a two-column log_softmax as a tiny-K dedicated template with straight-line loads, reductions, and log epilogue; the fix is NEW_PATTERN: add a guarded tiny-K log_softmax lowering for this amax/sub/exp/sum/log/sub idiom and only enable it when interleaved benchmarking beats the generic schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_a215204c9264"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
COLS = 2
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
CLASSIFICATION = "NEW_PATTERN"


@triton.jit
def _log_softmax_k2_kernel(
    x_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    mask = rows < 128

    x0 = tl.load(x_ptr + rows * x_s0, mask=mask, other=-float("inf")).to(tl.float32)
    x1 = tl.load(x_ptr + rows * x_s0 + x_s1, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.maximum(x0, x1)
    shifted0 = x0 - row_max
    shifted1 = x1 - row_max
    exp0 = tl.exp(shifted0)
    exp1 = tl.exp(shifted1)
    exp_sum = exp0 + exp1
    log_sum = tl.log(exp_sum)
    out0 = shifted0 - log_sum
    out1 = shifted1 - log_sum

    tl.store(out_ptr + rows * out_s0, out0, mask=mask)
    tl.store(out_ptr + rows * out_s0 + out_s1, out1, mask=mask)


def get_inputs() -> list[object]:
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


def _validate_input(addmm_72: torch.Tensor) -> None:
    if not addmm_72.is_cuda:
        raise RuntimeError("CUDA input is required")
    if addmm_72.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {addmm_72.dtype}")
    if tuple(addmm_72.shape) != OUT_SHAPE:
        raise ValueError(f"expected input shape {OUT_SHAPE}, got {tuple(addmm_72.shape)}")


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.float32)


def _launch_oracle(addmm_72: torch.Tensor, out: torch.Tensor, *, block_m: int) -> torch.Tensor:
    _validate_input(addmm_72)
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("output must be CUDA fp32")

    _log_softmax_k2_kernel[(triton.cdiv(ROWS, block_m),)](
        addmm_72,
        out,
        x_s0=addmm_72.stride(0),
        x_s1=addmm_72.stride(1),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        block_m=block_m,
        num_warps=4,
    )
    return out


def oracle_online_softmax(addmm_72: torch.Tensor, *, block_m: int = 32) -> torch.Tensor:
    out = _make_output(addmm_72.device)
    return _launch_oracle(addmm_72, out, block_m=block_m)


def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    (addmm_72,) = inputs
    if not isinstance(addmm_72, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm_72).__name__}")
    return oracle_online_softmax(addmm_72)


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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
