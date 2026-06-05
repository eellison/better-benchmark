"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Longformer duplicate-index scatter-add/layout/reduction tuple by emitting both required `[768, 2048]` transposed side-output layouts directly from the structured scatter coordinates and reducing those finalized layouts to the two `[768]` outputs, whereas tuned Inductor's full-scope lowering is already within the same runtime envelope for the required side-output materialization and reductions; Inductor cannot be assigned a local scatter-reduce optimization from this repro because the full-scope oracle benchmarks at parity after preserving both large side views and both reductions; the fix is BANDWIDTH_BOUND: record this as an at-floor structured scatter/layout case unless a faster full-scope oracle proves a real gap."""
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


# --- Oracle kernel(s) ---

BLOCKS = 24
CHUNKS = 3
HEADS = 12
BATCH = 2
WINDOW = 512
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = 1024 * BATCH
STORAGE = ROWS * HIDDEN
UPDATES = BLOCKS * CHUNKS * WINDOW * HEAD_DIM
ZERO_BLOCK = 1024
SCATTER_BLOCK = 1024
REDUCE_BLOCK = 2048
SCALE_BMM3 = 0.125


if triton is not None:

    @triton.jit
    def _zero_layouts_kernel(
        side_2_ptr,
        side_3_ptr,
        STORAGE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < STORAGE_
        zero = tl.full([BLOCK], 0.0, tl.float32)
        tl.store(side_2_ptr + offsets, zero, mask=active)
        tl.store(side_3_ptr + offsets, zero, mask=active)

    @triton.jit
    def _scatter_longformer_updates_kernel(
        bmm_2_ptr,
        bmm_3_ptr,
        side_2_ptr,
        side_3_ptr,
        UPDATES_: tl.constexpr,
        WINDOW_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        SCALE_BMM3_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < UPDATES_

        dim = offsets % HEAD_DIM_
        seq = (offsets // HEAD_DIM_) % WINDOW_
        chunk = (offsets // (HEAD_DIM_ * WINDOW_)) % 3
        block = offsets // (HEAD_DIM_ * WINDOW_ * 3)
        source_block = block * 3 + chunk

        target = block * HEAD_DIM_ + chunk * 393216 + seq * 1536 + dim
        bmm2_offsets = (source_block * HEAD_DIM_ + dim) * WINDOW_ + seq
        value2 = tl.load(bmm_2_ptr + bmm2_offsets, mask=active, other=0.0).to(tl.float32)
        value3 = tl.load(bmm_3_ptr + offsets, mask=active, other=0.0).to(tl.float32)

        tl.atomic_add(side_2_ptr + target, value2, sem="relaxed", mask=active)
        tl.atomic_add(side_3_ptr + target, value3 * SCALE_BMM3_, sem="relaxed", mask=active)

    @triton.jit
    def _reduce_layouts_kernel(
        side_2_ptr,
        side_3_ptr,
        sum_2_ptr,
        sum_3_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        hidden = tl.program_id(0)
        rows = tl.arange(0, BLOCK_ROWS)
        active = rows < ROWS_
        offsets = rows * HIDDEN_ + hidden
        value2 = tl.load(side_2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        value3 = tl.load(side_3_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(sum_2_ptr + hidden, tl.sum(value2, axis=0))
        tl.store(sum_3_ptr + hidden, tl.sum(value3, axis=0))


def _check_shape_params(
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
    _shape_param_10,
    _shape_param_11,
    _shape_param_12,
    _shape_param_13,
):
    assert list(_shape_param_0) == [24, 3, 64, 512, 1]
    assert list(_shape_param_1) == [24, 3, 512, 64, 1]
    assert list(_shape_param_2) == [2359296]
    assert list(_shape_param_3) == [2359296]
    assert list(_shape_param_4) == [24, 1024, 64]
    assert list(_shape_param_5) == [2, 12, 1024, 64]
    assert list(_shape_param_6) == [1024, 2, 768]
    assert list(_shape_param_7) == [24, 1024, 64]
    assert list(_shape_param_8) == [2, 12, 1024, 64]
    assert list(_shape_param_9) == [1024, 2, 768]
    assert list(_shape_param_10) == [768]
    assert list(_shape_param_11) == [2048, 768]
    assert list(_shape_param_12) == [768]
    assert list(_shape_param_13) == [2048, 768]


def _check_inputs(bmm_2: torch.Tensor, bmm_3: torch.Tensor):
    assert bmm_2.shape == (BLOCKS * CHUNKS, HEAD_DIM, WINDOW)
    assert bmm_3.shape == (BLOCKS * CHUNKS, WINDOW, HEAD_DIM)
    assert bmm_2.dtype == torch.float32
    assert bmm_3.dtype == torch.float32
    assert bmm_2.is_contiguous()
    assert bmm_3.is_contiguous()


def _oracle_triton(bmm_2: torch.Tensor, bmm_3: torch.Tensor):
    if triton is None:
        raise RuntimeError("triton is required for the Triton oracle")

    side_2 = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=bmm_2.device,
        dtype=torch.float32,
    )
    side_3 = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=bmm_2.device,
        dtype=torch.float32,
    )
    sum_2 = torch.empty((HIDDEN,), device=bmm_2.device, dtype=torch.float32)
    sum_3 = torch.empty((HIDDEN,), device=bmm_2.device, dtype=torch.float32)

    assert side_2.stride() == (1, HIDDEN)
    assert side_3.stride() == (1, HIDDEN)

    _zero_layouts_kernel[(triton.cdiv(STORAGE, ZERO_BLOCK),)](
        side_2,
        side_3,
        STORAGE_=STORAGE,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _scatter_longformer_updates_kernel[(triton.cdiv(UPDATES, SCATTER_BLOCK),)](
        bmm_2,
        bmm_3,
        side_2,
        side_3,
        UPDATES_=UPDATES,
        WINDOW_=WINDOW,
        HEAD_DIM_=HEAD_DIM,
        SCALE_BMM3_=SCALE_BMM3,
        BLOCK=SCATTER_BLOCK,
        num_warps=8,
    )
    _reduce_layouts_kernel[(HIDDEN,)](
        side_2,
        side_3,
        sum_2,
        sum_3,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS=REDUCE_BLOCK,
        num_warps=8,
    )
    return sum_2, side_2, sum_3, side_3


def _oracle_torch(
    bmm_2: torch.Tensor,
    bmm_3: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    _shape_param_8,
    _shape_param_9,
    _shape_param_10,
    _shape_param_11,
    _shape_param_12,
    _shape_param_13,
):
    del _shape_param_2, _shape_param_3
    updates_2 = (
        bmm_2.view(_shape_param_0)
        .permute(0, 1, 4, 3, 2)
        .permute(0, 1, 3, 4, 2)
        .squeeze(4)
        .contiguous()
        .view(-1)
    )
    updates_3 = bmm_3.view(_shape_param_1).squeeze(4).view(-1)
    indices = (
        torch.arange(STORAGE, device=bmm_2.device, dtype=torch.int64)
        .as_strided((24, 3, 512, 64), (64, 393216, 1536, 1))
        .contiguous()
        .view(-1)
    )

    scatter_2 = torch.zeros((STORAGE,), device=bmm_2.device, dtype=torch.float32)
    scatter_3 = torch.zeros((STORAGE,), device=bmm_2.device, dtype=torch.float32)
    scatter_2.index_put_((indices,), updates_2, accumulate=True)
    scatter_3.index_put_((indices,), updates_3, accumulate=True)

    logical_2 = (
        scatter_2.as_strided((24, 2, 512, 64), (64, 786432, 1536, 1))
        .view(_shape_param_7)
        .view(_shape_param_8)
        .permute(0, 2, 1, 3)
        .permute(1, 0, 2, 3)
        .view(_shape_param_9)
    )
    side_2 = logical_2.view(_shape_param_11).permute(1, 0)
    sum_2 = logical_2.sum([0, 1], keepdim=True).view(_shape_param_10)

    logical_3 = (
        scatter_3.as_strided((24, 2, 512, 64), (64, 786432, 1536, 1))
        .view(_shape_param_4)
        .view(_shape_param_5)
        .permute(0, 2, 1, 3)
        .permute(1, 0, 2, 3)
        .view(_shape_param_6)
        / 8.0
    )
    side_3 = logical_3.view(_shape_param_13).permute(1, 0)
    sum_3 = logical_3.sum([0, 1], keepdim=True).view(_shape_param_12)
    return sum_2, side_2, sum_3, side_3


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
    (
        bmm_2,
        bmm_3,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
    ) = inputs

    _check_inputs(bmm_2, bmm_3)
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
    )

    if bmm_2.device.type == "cuda" and triton is not None:
        return _oracle_triton(bmm_2, bmm_3)

    return _oracle_torch(
        bmm_2,
        bmm_3,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
        _shape_param_8,
        _shape_param_9,
        _shape_param_10,
        _shape_param_11,
        _shape_param_12,
        _shape_param_13,
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
