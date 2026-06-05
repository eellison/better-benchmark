"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DINOv2 class-token select-scatter layernorm-backward return tuple by directly gathering only valid token-0 contributors, sharing the rowwise reduction scalars, accumulating all returned channel reductions, and writing the required transposed side output without materializing the dense `[128, 1370, 768]` scatter/intermediate tensors, whereas Inductor currently materializes the zero-filled `select_scatter` producer and schedules its sibling reductions, layernorm-backward epilogues, reshape, and permute as generic work over the dense tensor; Inductor cannot do this today because scheduler/codegen does not model zero-fill select-scatter as a sparse gather-mask-reduce producer that can feed both reductions and required side-output stores; the fix is SCATTER_REDUCE: add a structured select-scatter/upsample-backward lowering that maps valid source lanes directly into shared reductions and side-output epilogues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


BATCH = 128
TOKENS = 1370
CHANNELS = 768
ROWS = BATCH * TOKENS
SIDE_NUMEL = ROWS * CHANNELS
OUTPUT_NUMEL = 4 * CHANNELS
BLOCK_ZERO = 1024
BLOCK_C = 1024

EXPECTED_SHAPE_PARAMS = (
    [BATCH, TOKENS, CHANNELS],
    [CHANNELS],
    [ROWS, CHANNELS],
    [CHANNELS],
)


def _check_inputs(inputs):
    if len(inputs) != 10:
        raise ValueError(f"expected 10 repro inputs, got {len(inputs)}")

    (
        tangents_1,
        primals_174,
        mul_108,
        div,
        addmm_47,
        primals_173,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    expected_tensors = (
        (tangents_1, (BATCH, CHANNELS), torch.float32, "tangents_1"),
        (primals_174, (CHANNELS,), torch.float32, "primals_174"),
        (mul_108, (BATCH, TOKENS, CHANNELS), torch.float32, "mul_108"),
        (div, (BATCH, TOKENS, 1), torch.float32, "div"),
        (addmm_47, (ROWS, CHANNELS), torch.float32, "addmm_47"),
        (primals_173, (CHANNELS,), torch.float32, "primals_173"),
    )
    for tensor, shape, dtype, name in expected_tensors:
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"unexpected {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    for idx, (actual, expected) in enumerate(
        zip((shape0, shape1, shape2, shape3), EXPECTED_SHAPE_PARAMS),
        start=6,
    ):
        if list(actual) != expected:
            raise ValueError(f"unexpected shape param {idx}: {actual} != {expected}")


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_buffers_kernel(
        side_ptr,
        sum_tangent_xhat_ptr,
        sum_tangent_ptr,
        sum_addmm_grad_ptr,
        sum_side_grad_ptr,
        total: tl.constexpr,
        side_numel: tl.constexpr,
        channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        zero = tl.zeros((BLOCK,), tl.float32)

        side_mask = offsets < side_numel
        tl.store(side_ptr + offsets, zero, mask=side_mask)

        tail = offsets - side_numel
        out0_mask = (tail >= 0) & (tail < channels)
        out1_mask = (tail >= channels) & (tail < 2 * channels)
        out2_mask = (tail >= 2 * channels) & (tail < 3 * channels)
        out4_mask = (tail >= 3 * channels) & (tail < 4 * channels)

        out0_offsets = tl.where(out0_mask, tail, 0)
        out1_offsets = tl.where(out1_mask, tail - channels, 0)
        out2_offsets = tl.where(out2_mask, tail - 2 * channels, 0)
        out4_offsets = tl.where(out4_mask, tail - 3 * channels, 0)

        tl.store(sum_tangent_xhat_ptr + out0_offsets, zero, mask=out0_mask)
        tl.store(sum_tangent_ptr + out1_offsets, zero, mask=out1_mask)
        tl.store(sum_addmm_grad_ptr + out2_offsets, zero, mask=out2_mask)
        tl.store(sum_side_grad_ptr + out4_offsets, zero, mask=out4_mask)

    @triton.jit
    def _token0_gather_reduce_kernel(
        tangents_ptr,
        gamma_ptr,
        xhat_ptr,
        div_ptr,
        addmm_ptr,
        side_scale_ptr,
        side_ptr,
        sum_tangent_xhat_ptr,
        sum_tangent_ptr,
        sum_addmm_grad_ptr,
        sum_side_grad_ptr,
        tangents_stride_b: tl.constexpr,
        gamma_stride_c: tl.constexpr,
        xhat_stride_b: tl.constexpr,
        xhat_stride_t: tl.constexpr,
        xhat_stride_c: tl.constexpr,
        div_stride_b: tl.constexpr,
        div_stride_t: tl.constexpr,
        addmm_stride_r: tl.constexpr,
        addmm_stride_c: tl.constexpr,
        side_scale_stride_c: tl.constexpr,
        side_stride_r: tl.constexpr,
        side_stride_c: tl.constexpr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < channels

        tangents = tl.load(
            tangents_ptr + batch * tangents_stride_b + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            gamma_ptr + cols * gamma_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        xhat = tl.load(
            xhat_ptr
            + batch * xhat_stride_b
            + cols * xhat_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(div_ptr + batch * div_stride_b).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + (batch * tokens) * addmm_stride_r + cols * addmm_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        side_scale = tl.load(
            side_scale_ptr + cols * side_scale_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        weighted = tangents * gamma
        row_sum = tl.sum(weighted, axis=0)
        row_dot = tl.sum(weighted * xhat, axis=0)
        grad = scale * (weighted * channels - row_sum - xhat * row_dot)
        side_value = grad * side_scale

        side_offsets = (batch * tokens) * side_stride_r + cols * side_stride_c
        tl.store(side_ptr + side_offsets, side_value, mask=mask)

        tl.atomic_add(
            sum_tangent_xhat_ptr + cols,
            tangents * xhat,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            sum_tangent_ptr + cols,
            tangents,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            sum_addmm_grad_ptr + cols,
            grad * addmm,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            sum_side_grad_ptr + cols,
            side_value,
            sem="relaxed",
            mask=mask,
        )


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
        raise RuntimeError("triton is required for this oracle")
    _check_inputs(inputs)
    (
        tangents_1,
        primals_174,
        mul_108,
        div,
        addmm_47,
        primals_173,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    if tangents_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    sum_tangent_xhat = torch.empty((CHANNELS,), device=tangents_1.device, dtype=torch.float32)
    sum_tangent = torch.empty((CHANNELS,), device=tangents_1.device, dtype=torch.float32)
    sum_addmm_grad = torch.empty((CHANNELS,), device=tangents_1.device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=tangents_1.device, dtype=torch.float32)
    sum_side_grad = torch.empty((CHANNELS,), device=tangents_1.device, dtype=torch.float32)

    total = SIDE_NUMEL + OUTPUT_NUMEL
    _zero_buffers_kernel[(triton.cdiv(total, BLOCK_ZERO),)](
        side_base,
        sum_tangent_xhat,
        sum_tangent,
        sum_addmm_grad,
        sum_side_grad,
        total,
        SIDE_NUMEL,
        CHANNELS,
        BLOCK=BLOCK_ZERO,
        num_warps=4,
    )
    _token0_gather_reduce_kernel[(BATCH,)](
        tangents_1,
        primals_174,
        mul_108,
        div,
        addmm_47,
        primals_173,
        side_base,
        sum_tangent_xhat,
        sum_tangent,
        sum_addmm_grad,
        sum_side_grad,
        tangents_stride_b=tangents_1.stride(0),
        gamma_stride_c=primals_174.stride(0),
        xhat_stride_b=mul_108.stride(0),
        xhat_stride_t=mul_108.stride(1),
        xhat_stride_c=mul_108.stride(2),
        div_stride_b=div.stride(0),
        div_stride_t=div.stride(1),
        addmm_stride_r=addmm_47.stride(0),
        addmm_stride_c=addmm_47.stride(1),
        side_scale_stride_c=primals_173.stride(0),
        side_stride_r=side_base.stride(0),
        side_stride_c=side_base.stride(1),
        channels=CHANNELS,
        tokens=TOKENS,
        BLOCK=BLOCK_C,
        num_warps=8,
    )

    return (
        sum_tangent_xhat,
        sum_tangent,
        sum_addmm_grad,
        side_base.t(),
        sum_side_grad,
    )


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
