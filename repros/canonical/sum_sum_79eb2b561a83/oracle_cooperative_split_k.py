"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle cooperatively splits the shared `N,H,W` channel reductions for both sibling sums, finalizes the `[48]` scale-gradient side output, and uses those same finalized summaries to write the full `[512,48,56,56]` batch-norm-backward tensor, whereas Inductor currently schedules the sibling reductions and dependent pointwise epilogue as generic reduction/pointwise work without one coordinated full-scope split-K plan; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that finalizes small channel summaries before feeding a dependent full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward channel-reduction lowering that splits the reduced `N,H,W` domain, combines both sibling summaries, and fuses the tensor/vector epilogues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

N = 512
C = 48
H = 56
W = 56
HW = H * W
NUMEL = N * C * HW
SCALE = 6.228077168367346e-07

OUT0_SHAPE = (N, C, H, W)
OUT0_STRIDE = (C * HW, HW, W, 1)
MEAN_SHAPE = (1, C, 1, 1)
MEAN_STRIDE = (C, 1, 1, 1)
OUT1_SHAPE = (C,)
OUT1_STRIDE = (1,)

REDUCE_BLOCK_K = 2048
EPILOGUE_BLOCK_ELEMS = 256


if triton is not None:

    @triton.jit
    def _atomic_bn_sums_kernel(
        grad_ptr,
        activation_ptr,
        mean_ptr,
        grad_sum_ptr,
        centered_sum_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_K_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        c = tl.program_id(0)
        k = tl.program_id(1) * BLOCK_K_ + tl.arange(0, BLOCK_K_)
        mask = k < TOTAL_K_

        n = k // HW_
        hw = k - n * HW_
        offsets = n * C_ * HW_ + c * HW_ + hw

        mean = tl.load(mean_ptr + c).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32) - mean
        active_grad = tl.where(mask, grad, 0.0)
        active_product = tl.where(mask, grad * centered, 0.0)

        tl.atomic_add(grad_sum_ptr + c, tl.sum(active_grad, axis=0), sem="relaxed")
        tl.atomic_add(centered_sum_ptr + c, tl.sum(active_product, axis=0), sem="relaxed")

    @triton.jit
    def _bn_backward_epilogue_kernel(
        grad_ptr,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        grad_sum_ptr,
        centered_sum_ptr,
        out0_ptr,
        out1_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_ELEMS_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
        mask = offsets < NUMEL_

        c = (offsets // HW_) % C_
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        grad_sum = tl.load(grad_sum_ptr + c, mask=mask, other=0.0).to(tl.float32)
        centered_sum = tl.load(centered_sum_ptr + c, mask=mask, other=0.0).to(tl.float32)
        mean_term = grad_sum * SCALE_
        variance_term = centered_sum * SCALE_ * invstd * invstd

        centered = activation - mean
        out = (grad - centered * variance_term - mean_term) * (invstd * weight)
        tl.store(out0_ptr + offsets, out, mask=mask)

        hw = offsets % HW_
        n = offsets // (C_ * HW_)
        tl.store(out1_ptr + c, centered_sum * invstd, mask=mask & (n == 0) & (hw == 0))


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    getitem_258, arg219_1, arg534_1, arg220_1, arg18_1 = inputs

    getitem_258 = _require_f32_tensor("getitem_258", getitem_258, OUT0_SHAPE, OUT0_STRIDE)
    arg219_1 = _require_f32_tensor("arg219_1", arg219_1, OUT0_SHAPE, OUT0_STRIDE)
    arg534_1 = _require_f32_tensor("arg534_1", arg534_1, MEAN_SHAPE, MEAN_STRIDE)
    arg220_1 = _require_f32_tensor("arg220_1", arg220_1, OUT1_SHAPE, OUT1_STRIDE)
    arg18_1 = _require_f32_tensor("arg18_1", arg18_1, OUT1_SHAPE, OUT1_STRIDE)

    device = getitem_258.device
    if any(t.device != device for t in (arg219_1, arg534_1, arg220_1, arg18_1)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return getitem_258, arg219_1, arg534_1, arg220_1, arg18_1


@oracle_impl(hardware="H100", shapes="(T([512, 48, 56, 56], f32), T([512, 48, 56, 56], f32), T([1, 48, 1, 1], f32), T([48], f32), T([48], f32))")
def oracle_forward(inputs):
    """Run the oracle computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    getitem_258, arg219_1, arg534_1, arg220_1, arg18_1 = _validate_inputs(inputs)

    grad_sum = torch.zeros(OUT1_SHAPE, device=getitem_258.device, dtype=torch.float32)
    centered_sum = torch.zeros(OUT1_SHAPE, device=getitem_258.device, dtype=torch.float32)
    out0 = torch.empty_strided(OUT0_SHAPE, OUT0_STRIDE, device=getitem_258.device, dtype=torch.float32)
    out1 = torch.empty_strided(OUT1_SHAPE, OUT1_STRIDE, device=getitem_258.device, dtype=torch.float32)

    _atomic_bn_sums_kernel[(C, triton.cdiv(N * HW, REDUCE_BLOCK_K))](
        getitem_258,
        arg219_1,
        arg534_1,
        grad_sum,
        centered_sum,
        C_=C,
        HW_=HW,
        TOTAL_K_=N * HW,
        BLOCK_K_=REDUCE_BLOCK_K,
        num_warps=4,
    )

    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        getitem_258,
        arg219_1,
        arg534_1,
        arg220_1,
        arg18_1,
        grad_sum,
        centered_sum,
        out0,
        out1,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS_=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    return out0, out1


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
