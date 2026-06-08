"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer BN-backward-style scope with shared `sum(x * grad)` and `sum(x)` channel reductions, one fused f32 summary finalizer, the returned channel vector, and the full tensor epilogue, whereas coordinate-descent Inductor already emits the same split channel-reduction structure and f32 epilogue with only tiny extra finalizer launch work; Inductor cannot materially improve this local repro through a narrower scheduler-fusion change because runtime is dominated by the required two activation/gradient reads, add-input read, and full output store; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope reduction-plus-epilogue case unless a broader memory-traffic or launch-overhead improvement moves both paths."""
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

N = 128
C = 192
H = 28
W = 28
HW = H * W
TOTAL = N * C * HW
REDUCE_GROUPS = 4
GROUP_N = N // REDUCE_GROUPS
GROUP_REDUCE = GROUP_N * HW
INV_REDUCE = 9.964923469387754e-06
REDUCE_BLOCK = 2048
POINTWISE_BLOCK = 1024

ACT_SHAPE = (N, C, H, W)
ACT_STRIDE = (C * HW, HW, W, 1)
CHANNEL_SHAPE = (C,)
CHANNEL_STRIDE = (1,)

CLASSIFICATION = "BANDWIDTH_BOUND"


if triton is not None:

    @triton.jit
    def _partial_channel_sums_kernel(
        x_ptr,
        grad_ptr,
        partial_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        GROUP_N_: tl.constexpr,
        GROUP_REDUCE_: tl.constexpr,
        REDUCE_GROUPS_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        c = tl.program_id(0)
        group = tl.program_id(1)
        r_offsets = tl.arange(0, BLOCK_R)
        sum_x_grad = tl.zeros([BLOCK_R], dtype=tl.float32)
        sum_x = tl.zeros([BLOCK_R], dtype=tl.float32)

        for base in tl.range(0, GROUP_REDUCE_, BLOCK_R):
            r = base + r_offsets
            mask = r < GROUP_REDUCE_
            n = group * GROUP_N_ + (r // HW_)
            hw = r - (r // HW_) * HW_
            offsets = n * C_ * HW_ + c * HW_ + hw

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            prod = x * grad
            sum_x_grad = tl.where(mask, sum_x_grad + prod, sum_x_grad)
            sum_x = tl.where(mask, sum_x + x, sum_x)

        partial_offset = c * REDUCE_GROUPS_ + group
        plane_stride = C_ * REDUCE_GROUPS_
        tl.store(partial_ptr + partial_offset, tl.sum(sum_x_grad, axis=0))
        tl.store(partial_ptr + plane_stride + partial_offset, tl.sum(sum_x, axis=0))

    @triton.jit
    def _finalize_channel_sums_kernel(
        partial_ptr,
        gamma_ptr,
        stats_ptr,
        out0_ptr,
        C_: tl.constexpr,
        REDUCE_GROUPS_: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        c = tl.program_id(0)
        groups = tl.arange(0, BLOCK_G)
        mask = groups < REDUCE_GROUPS_
        plane_stride = C_ * REDUCE_GROUPS_
        partial_offset = c * REDUCE_GROUPS_ + groups

        partial_x_grad = tl.load(partial_ptr + partial_offset, mask=mask, other=0.0).to(tl.float32)
        partial_x = tl.load(partial_ptr + plane_stride + partial_offset, mask=mask, other=0.0).to(tl.float32)
        sum_x_grad = tl.sum(partial_x_grad, axis=0).to(tl.float32)
        sum_x = tl.sum(partial_x, axis=0).to(tl.float32)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)

        tl.store(stats_ptr + c, sum_x_grad)
        tl.store(stats_ptr + C_ + c, sum_x)
        tl.store(out0_ptr + c, sum_x_grad * gamma)

    @triton.jit
    def _write_epilogue_kernel(
        add_ptr,
        x_ptr,
        grad_ptr,
        stats_ptr,
        gamma_ptr,
        beta_ptr,
        out1_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_: tl.constexpr,
        INV_REDUCE_: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
        mask = offsets < TOTAL_
        c = (offsets // HW_) % C_

        add_value = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x_grad = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + c, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.load(stats_ptr + C_ + c, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + c, mask=mask, other=0.0).to(tl.float32)

        scale_const = tl.full([1], INV_REDUCE_, tl.float32)
        mean_x_grad = sum_x_grad * scale_const
        gamma_squared = gamma * gamma
        correction = mean_x_grad * gamma_squared
        grad_correction = grad * correction
        centered = x - grad_correction
        mean_x = sum_x * scale_const
        centered = centered - mean_x
        output_scale = gamma * beta
        result = centered * output_scale
        result = add_value + result
        tl.store(out1_ptr + offsets, result, mask=mask)


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

    getitem_153, arg261_1, arg105_1, arg11_1, add_60 = inputs
    getitem_153 = _require_f32_tensor("getitem_153", getitem_153, ACT_SHAPE, ACT_STRIDE)
    arg261_1 = _require_f32_tensor("arg261_1", arg261_1, ACT_SHAPE, ACT_STRIDE)
    arg105_1 = _require_f32_tensor("arg105_1", arg105_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    arg11_1 = _require_f32_tensor("arg11_1", arg11_1, CHANNEL_SHAPE, CHANNEL_STRIDE)
    add_60 = _require_f32_tensor("add_60", add_60, ACT_SHAPE, ACT_STRIDE)

    device = getitem_153.device
    if any(t.device != device for t in (arg261_1, arg105_1, arg11_1, add_60)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return getitem_153, arg261_1, arg105_1, arg11_1, add_60


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_visformer_bn_backward.py")

    getitem_153, arg261_1, arg105_1, arg11_1, add_60 = _validate_inputs(inputs)
    partial = torch.empty_strided(
        (2, C, REDUCE_GROUPS),
        (C * REDUCE_GROUPS, REDUCE_GROUPS, 1),
        device=getitem_153.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(
        (2, C),
        (C, 1),
        device=getitem_153.device,
        dtype=torch.float32,
    )
    out0 = torch.empty_strided(CHANNEL_SHAPE, CHANNEL_STRIDE, device=getitem_153.device, dtype=torch.float32)
    out1 = torch.empty_strided(ACT_SHAPE, ACT_STRIDE, device=getitem_153.device, dtype=torch.float32)

    _partial_channel_sums_kernel[(C, REDUCE_GROUPS)](
        getitem_153,
        arg261_1,
        partial,
        C_=C,
        HW_=HW,
        GROUP_N_=GROUP_N,
        GROUP_REDUCE_=GROUP_REDUCE,
        REDUCE_GROUPS_=REDUCE_GROUPS,
        BLOCK_R=REDUCE_BLOCK,
        num_warps=16,
        num_stages=1,
    )
    _finalize_channel_sums_kernel[(C,)](
        partial,
        arg105_1,
        stats,
        out0,
        C_=C,
        REDUCE_GROUPS_=REDUCE_GROUPS,
        BLOCK_G=REDUCE_GROUPS,
        num_warps=1,
        num_stages=1,
    )
    _write_epilogue_kernel[(triton.cdiv(TOTAL, POINTWISE_BLOCK),)](
        add_60,
        getitem_153,
        arg261_1,
        stats,
        arg105_1,
        arg11_1,
        out1,
        C_=C,
        HW_=HW,
        TOTAL_=TOTAL,
        INV_REDUCE_=INV_REDUCE,
        BLOCK_E=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=1,
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
