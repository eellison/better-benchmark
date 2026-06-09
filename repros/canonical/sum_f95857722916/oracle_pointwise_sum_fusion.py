"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete dropout-scaled exact-erf GELU-gradient pointwise tensor, writes the returned transposed-view backing storage, and accumulates the first column-sum partials in the same Triton pass, whereas Inductor materializes that pointwise tensor and rereads it through a separate two-stage reduction; Inductor cannot do this today because the scheduler does not emit a multi-output pointwise-plus-reduction kernel when the pointwise result is also returned as a layout view; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse a returned pointwise producer with its reduction consumer while preserving the returned storage layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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

M = 16384
N = 3072
ROWS_PER_PARTIAL = 64
ROW_BLOCK = 64
NUM_PARTIALS = 256
POINTWISE_COL_BLOCK = 32
FINAL_COL_BLOCK = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _pointwise_store_and_partial_sum_kernel(
        mm_ptr,
        mask_ptr,
        gelu_arg_ptr,
        out_ptr,
        partial_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        COL_BLOCK: tl.constexpr,
        ROW_BLOCK_: tl.constexpr,
        ROWS_PER_PARTIAL_: tl.constexpr,
    ):
        col_pid = tl.program_id(0)
        partial_pid = tl.program_id(1)
        cols = col_pid * COL_BLOCK + tl.arange(0, COL_BLOCK)[:, None]
        r = tl.arange(0, ROW_BLOCK_)[None, :]
        rows = partial_pid * ROWS_PER_PARTIAL_ + r
        valid = (cols < N_) & (rows < M_) & (r < ROWS_PER_PARTIAL_)
        offsets = rows * N_ + cols

        tmp0 = tl.load(mm_ptr + offsets, valid, eviction_policy="evict_first", other=0.0)
        tmp1 = tl.load(mask_ptr + offsets, valid, eviction_policy="evict_first", other=0).to(tl.int1)
        tmp6 = tl.load(gelu_arg_ptr + offsets, valid, eviction_policy="evict_first", other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.full([1, 1], 1.1111111111111112, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp0 * tmp4
        tmp7 = tl.full([1, 1], 0.7071067811865476, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = libdevice.erf(tmp8)
        tmp10 = tl.full([1, 1], 1.0, tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full([1, 1], 0.5, tl.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp6
        tmp15 = tl.full([1, 1], -0.5, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = libdevice.exp(tmp16)
        tmp18 = tl.full([1, 1], 0.3989422804014327, tl.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tmp13 + tmp20
        tmp22 = tmp5 * tmp21

        tl.store(out_ptr + offsets, tmp22, valid)
        sum_terms = tl.where(valid, tmp22, 0.0)
        partial = tl.sum(sum_terms, 1)[:, None]
        tl.store(partial_ptr + partial_pid * N_ + cols, partial, cols < N_)

    @triton.jit
    def _finish_sum_kernel(
        partial_ptr,
        sum_ptr,
        N_: tl.constexpr,
        NUM_PARTIALS_: tl.constexpr,
        COL_BLOCK: tl.constexpr,
        PARTIAL_BLOCK: tl.constexpr,
    ):
        col_pid = tl.program_id(0)
        cols = col_pid * COL_BLOCK + tl.arange(0, COL_BLOCK)[:, None]
        partials = tl.arange(0, PARTIAL_BLOCK)[None, :]
        valid = (cols < N_) & (partials < NUM_PARTIALS_)
        values = tl.load(
            partial_ptr + partials * N_ + cols,
            valid,
            eviction_policy="evict_first",
            other=0.0,
        )
        total = tl.sum(tl.where(valid, values, 0.0), 1)[:, None].to(tl.float32)
        tl.store(sum_ptr + cols, total, cols < N_)


def _validate_inputs(inputs):
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with Inductor libdevice helpers is required")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm_136, arg115_1, arg114_1, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(mm_136, torch.Tensor) or not isinstance(arg115_1, torch.Tensor):
        raise TypeError("first two inputs must be tensors")
    if not isinstance(arg114_1, torch.Tensor):
        raise TypeError("third input must be a tensor")
    if tuple(mm_136.shape) != (M, N) or tuple(mm_136.stride()) != (N, 1):
        raise ValueError(f"unexpected mm_136 layout: shape={tuple(mm_136.shape)} stride={tuple(mm_136.stride())}")
    if tuple(arg115_1.shape) != (128, 128, N) or tuple(arg115_1.stride()) != (128 * N, N, 1):
        raise ValueError(f"unexpected arg115_1 layout: shape={tuple(arg115_1.shape)} stride={tuple(arg115_1.stride())}")
    if tuple(arg114_1.shape) != (M, N) or tuple(arg114_1.stride()) != (N, 1):
        raise ValueError(f"unexpected arg114_1 layout: shape={tuple(arg114_1.shape)} stride={tuple(arg114_1.stride())}")
    if mm_136.dtype != torch.float32 or arg114_1.dtype != torch.float32:
        raise TypeError("oracle expects f32 matrix inputs")
    if arg115_1.dtype != torch.bool:
        raise TypeError("oracle expects a bool dropout mask")
    if not mm_136.is_cuda or not arg115_1.is_cuda or not arg114_1.is_cuda:
        raise ValueError("oracle expects CUDA tensor inputs")
    if list(shape0) != [128, 128, N] or list(shape1) != [128, 128, N]:
        raise ValueError("unexpected 3D view shape parameters")
    if list(shape2) != [M, N] or list(shape3) != [N]:
        raise ValueError("unexpected output view shape parameters")
    return mm_136, arg115_1, arg114_1


@oracle_impl(hardware="H100", shapes="(T([16384, 3072], f32), T([128, 128, 3072], b8), T([16384, 3072], f32), S([128, 128, 3072]), S([128, 128, 3072]), S([16384, 3072]), S([3072]))")
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
    mm_136, arg115_1, arg114_1 = _validate_inputs(inputs)
    output_base = torch.empty_strided(
        (M, N),
        (N, 1),
        device=mm_136.device,
        dtype=torch.float32,
    )
    output = output_base.as_strided((N, M), (1, N))
    partials = torch.empty_strided(
        (NUM_PARTIALS, N),
        (N, 1),
        device=mm_136.device,
        dtype=torch.float32,
    )
    sum_base = torch.empty_strided(
        (1, N),
        (N, 1),
        device=mm_136.device,
        dtype=torch.float32,
    )
    sum_output = sum_base.as_strided((N,), (1,))

    _pointwise_store_and_partial_sum_kernel[(triton.cdiv(N, POINTWISE_COL_BLOCK), NUM_PARTIALS)](
        mm_136,
        arg115_1,
        arg114_1,
        output_base,
        partials,
        M_=M,
        N_=N,
        COL_BLOCK=POINTWISE_COL_BLOCK,
        ROW_BLOCK_=ROW_BLOCK,
        ROWS_PER_PARTIAL_=ROWS_PER_PARTIAL,
        num_warps=8,
    )
    _finish_sum_kernel[(triton.cdiv(N, FINAL_COL_BLOCK),)](
        partials,
        sum_base,
        N_=N,
        NUM_PARTIALS_=NUM_PARTIALS,
        COL_BLOCK=FINAL_COL_BLOCK,
        PARTIAL_BLOCK=256,
        num_warps=2,
    )
    return output, sum_output


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
