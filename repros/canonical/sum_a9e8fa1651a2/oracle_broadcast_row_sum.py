"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete dropout-scaled attention row-sum backward epilogue as one flat Triton row reduction with the final broadcast-mask where and output view included, whereas Inductor already emits the same full-scope persistent reduction without materializing intermediates; Inductor cannot substantially improve this repro with scheduling alone because the fused kernel is dominated by contiguous reads of two f32 tensors plus mask reads and one f32 output write; the fix is BANDWIDTH_BOUND: keep the current single-kernel lowering and treat further changes as tile retuning rather than a missing fusion pattern."""
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


if triton is not None:

    @triton.jit
    def _flat_row_sum_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        grad_ptr,
        where_mask_ptr,
        fill_ptr,
        out_ptr,
        XBLOCK: tl.constexpr,
    ):
        rblock: tl.constexpr = 512
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        cols = tl.arange(0, rblock)[None, :]
        offsets = cols + rblock * xindex
        query = xindex % 512
        batch = xindex // 12288

        bmm = tl.load(bmm_ptr + offsets, eviction_policy="evict_first")
        keep = tl.load(
            dropout_mask_ptr + offsets,
            eviction_policy="evict_first",
        ).to(tl.int1)
        grad = tl.load(grad_ptr + offsets, eviction_policy="evict_first")

        keep_f32 = keep.to(tl.float32)
        scale = tl.full([1, 1], 1.1111111111111112, tl.float32)
        scaled_keep = keep_f32 * scale
        mul1 = bmm * scaled_keep
        mul2 = mul1 * grad
        row_sum = tl.sum(tl.broadcast_to(mul2, [XBLOCK, rblock]), 1)[:, None].to(tl.float32)

        neg_grad = -grad
        value = tl.fma(neg_grad, row_sum, mul2)

        where_offsets = cols + 512 * query + 262144 * batch
        where_mask = tl.load(where_mask_ptr + where_offsets, eviction_policy="evict_last").to(tl.int1)
        fill = tl.load(fill_ptr + 0)
        fill = tl.broadcast_to(fill, [1, 1])
        result = tl.where(where_mask, fill, value)

        tl.store(out_ptr + offsets, result)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")

    bmm_89, arg219_1, arg218_1, full_2, full_1, shape0, shape1 = inputs
    if not isinstance(bmm_89, torch.Tensor) or bmm_89.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if bmm_89.dtype != torch.float32 or arg218_1.dtype != torch.float32 or full_1.dtype != torch.float32:
        raise ValueError("expected f32 data tensors and f32 fill scalar")
    if arg219_1.dtype != torch.bool or full_2.dtype != torch.bool:
        raise ValueError("expected bool masks")
    if tuple(shape0) != (8, 24, 512, 512) or tuple(shape1) != (192, 512, 512):
        raise ValueError(f"unexpected shape params: {shape0}, {shape1}")
    if tuple(bmm_89.shape) != (192, 512, 512):
        raise ValueError(f"unexpected bmm shape: {tuple(bmm_89.shape)}")
    if tuple(arg219_1.shape) != (8, 24, 512, 512):
        raise ValueError(f"unexpected dropout mask shape: {tuple(arg219_1.shape)}")
    if tuple(arg218_1.shape) != (8, 24, 512, 512):
        raise ValueError(f"unexpected grad shape: {tuple(arg218_1.shape)}")
    if tuple(full_2.shape) != (8, 1, 512, 512):
        raise ValueError(f"unexpected where mask shape: {tuple(full_2.shape)}")
    if tuple(full_1.shape) != ():
        raise ValueError(f"unexpected fill shape: {tuple(full_1.shape)}")
    expected_strides = {
        "bmm": (262144, 512, 1),
        "dropout": (6291456, 262144, 512, 1),
        "grad": (6291456, 262144, 512, 1),
        "where": (262144, 262144, 512, 1),
    }
    if tuple(bmm_89.stride()) != expected_strides["bmm"]:
        raise ValueError(f"unexpected bmm stride: {tuple(bmm_89.stride())}")
    if tuple(arg219_1.stride()) != expected_strides["dropout"]:
        raise ValueError(f"unexpected dropout mask stride: {tuple(arg219_1.stride())}")
    if tuple(arg218_1.stride()) != expected_strides["grad"]:
        raise ValueError(f"unexpected grad stride: {tuple(arg218_1.stride())}")
    if tuple(full_2.stride()) != expected_strides["where"]:
        raise ValueError(f"unexpected where mask stride: {tuple(full_2.stride())}")

    return bmm_89, arg219_1, arg218_1, full_2, full_1


@oracle_impl(hardware="H100", shapes="(T([192, 512, 512], f32), T([8, 24, 512, 512], b8), T([8, 24, 512, 512], f32), T([8, 1, 512, 512], b8), T([], f32), S([8, 24, 512, 512]), S([192, 512, 512]))")
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
    bmm_89, arg219_1, arg218_1, full_2, full_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        (192, 512, 512),
        (262144, 512, 1),
        device=bmm_89.device,
        dtype=torch.float32,
    )

    _flat_row_sum_kernel[(98304,)](
        bmm_89,
        arg219_1,
        arg218_1,
        full_2,
        full_1,
        out,
        XBLOCK=1,
        num_warps=4,
        num_stages=1,
    )
    return out


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
