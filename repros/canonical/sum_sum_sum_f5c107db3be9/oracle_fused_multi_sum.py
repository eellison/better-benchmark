"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Repro.forward scope by materializing the required contiguous bmm layout clone, returning its transposed view, and reducing all eleven input matrices plus the cloned bmm matrix through one shared 128-row split reduction and one shared final add kernel, whereas Inductor currently emits the layout clone, twelve separate first-stage column-sum kernels, and a final add/reduction kernel; Inductor cannot do this today because the scheduler does not group same-domain sibling reductions over different sources into one multi-accumulator reduction plan that preserves the live layout clone and final output contract; the fix is SCHEDULER_FUSION: teach reduction scheduling to co-schedule equal-shape column reductions from multiple inputs and fuse their final left-associative add epilogue."""
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


MATRIX_ROWS = 4096
MATRIX_COLS = 4096
NUM_BMM_ELEMENTS = MATRIX_ROWS * MATRIX_COLS
REDUCTION_CHUNK = 128
NUM_CHUNKS = MATRIX_ROWS // REDUCTION_CHUNK
NUM_SOURCES = 12


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _clone_bmm_layout_kernel(
        bmm_ptr,
        clone_ptr,
        TOTAL: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = linear < TOTAL

        x0 = linear % 64
        x1 = (linear // 64) % 64
        x2 = (linear // 4096) % 512
        x3 = linear // 2097152
        src_offset = x0 + 64 * x2 + 32768 * x1 + 2097152 * x3
        value = tl.load(bmm_ptr + src_offset, mask=mask, other=0.0)
        tl.store(clone_ptr + linear, value, mask=mask)

    @triton.jit
    def _store_source_partial(
        src_ptr,
        partial_ptr,
        source: tl.constexpr,
        chunk: tl.tensor,
        cols: tl.tensor,
        col_mask: tl.tensor,
        MATRIX_COLS_: tl.constexpr,
        REDUCTION_CHUNK_: tl.constexpr,
        NUM_CHUNKS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = chunk * REDUCTION_CHUNK_ + tl.arange(0, REDUCTION_CHUNK_)[:, None]
        offsets = rows * MATRIX_COLS_ + cols[None, :]
        values = tl.load(src_ptr + offsets, mask=col_mask[None, :], other=0.0)
        reduced = tl.sum(values, axis=0).to(tl.float32)
        partial_offset = source * NUM_CHUNKS_ * MATRIX_COLS_ + chunk * MATRIX_COLS_ + cols
        tl.store(partial_ptr + partial_offset, reduced, mask=col_mask)

    @triton.jit
    def _multi_source_partial_sum_kernel(
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        clone_ptr,
        partial_ptr,
        MATRIX_COLS_: tl.constexpr,
        REDUCTION_CHUNK_: tl.constexpr,
        NUM_CHUNKS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < MATRIX_COLS_

        _store_source_partial(in0, partial_ptr, 0, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in1, partial_ptr, 1, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in2, partial_ptr, 2, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in3, partial_ptr, 3, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in4, partial_ptr, 4, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in5, partial_ptr, 5, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in6, partial_ptr, 6, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in7, partial_ptr, 7, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in8, partial_ptr, 8, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in9, partial_ptr, 9, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(in10, partial_ptr, 10, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)
        _store_source_partial(clone_ptr, partial_ptr, 11, chunk, cols, col_mask, MATRIX_COLS_, REDUCTION_CHUNK_, NUM_CHUNKS_, BLOCK_N)

    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        out_ptr,
        MATRIX_COLS_: tl.constexpr,
        NUM_CHUNKS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < MATRIX_COLS_
        chunks = tl.arange(0, NUM_CHUNKS_)[:, None]

        off0 = 0 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off1 = 1 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off2 = 2 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off3 = 3 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off4 = 4 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off5 = 5 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off6 = 6 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off7 = 7 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off8 = 8 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off9 = 9 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off10 = 10 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]
        off11 = 11 * NUM_CHUNKS_ * MATRIX_COLS_ + chunks * MATRIX_COLS_ + cols[None, :]

        s0 = tl.sum(tl.load(partial_ptr + off0, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s1 = tl.sum(tl.load(partial_ptr + off1, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s2 = tl.sum(tl.load(partial_ptr + off2, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s3 = tl.sum(tl.load(partial_ptr + off3, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s4 = tl.sum(tl.load(partial_ptr + off4, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s5 = tl.sum(tl.load(partial_ptr + off5, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s6 = tl.sum(tl.load(partial_ptr + off6, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s7 = tl.sum(tl.load(partial_ptr + off7, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s8 = tl.sum(tl.load(partial_ptr + off8, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s9 = tl.sum(tl.load(partial_ptr + off9, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s10 = tl.sum(tl.load(partial_ptr + off10, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)
        s11 = tl.sum(tl.load(partial_ptr + off11, mask=col_mask[None, :], other=0.0), axis=0).to(tl.float32)

        # Preserve the captured left-associative add order:
        # sums(view_30..view_330) followed by the cloned bmm layout sum.
        total = s0 + s1
        total = total + s2
        total = total + s3
        total = total + s4
        total = total + s5
        total = total + s6
        total = total + s7
        total = total + s8
        total = total + s9
        total = total + s10
        total = total + s11
        tl.store(out_ptr + cols, total, mask=col_mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 27:
        raise ValueError(f"expected 27 inputs, got {len(inputs)}")

    matrices = inputs[:11]
    bmm = inputs[11]
    for index, tensor in enumerate(matrices):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"input {index} is not a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tensor.dtype != torch.float32:
            raise ValueError(f"input {index} expected torch.float32, got {tensor.dtype}")
        if tuple(tensor.shape) != (MATRIX_ROWS, MATRIX_COLS):
            raise ValueError(f"input {index} has unexpected shape {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != (MATRIX_COLS, 1):
            raise ValueError(f"input {index} has unexpected stride {tensor.stride()}")

    if not isinstance(bmm, torch.Tensor):
        raise TypeError("input 11 is not a tensor")
    if bmm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if bmm.dtype != torch.float32:
        raise ValueError(f"input 11 expected torch.float32, got {bmm.dtype}")
    if tuple(bmm.shape) != (512, 512, 64):
        raise ValueError(f"input 11 has unexpected shape {tuple(bmm.shape)}")
    if tuple(bmm.stride()) != (32768, 64, 1):
        raise ValueError(f"input 11 has unexpected stride {bmm.stride()}")

    return (*matrices, bmm)


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward().

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two outputs. Output 0 is the stride-(1, 4096) transpose view over the
    materialized contiguous bmm clone; output 1 is the fp32 left-associative sum
    of the eleven column reductions plus the cloned bmm column reduction.
    """
    in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, bmm = _validate_inputs(inputs)

    clone_base = torch.empty_strided(
        (MATRIX_ROWS, MATRIX_COLS),
        (MATRIX_COLS, 1),
        device=bmm.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (NUM_SOURCES, NUM_CHUNKS, MATRIX_COLS),
        (NUM_CHUNKS * MATRIX_COLS, MATRIX_COLS, 1),
        device=bmm.device,
        dtype=torch.float32,
    )
    summed = torch.empty_strided((MATRIX_COLS,), (1,), device=bmm.device, dtype=torch.float32)

    clone_block = 4096
    _clone_bmm_layout_kernel[(triton.cdiv(NUM_BMM_ELEMENTS, clone_block),)](
        bmm,
        clone_base,
        TOTAL=NUM_BMM_ELEMENTS,
        XBLOCK=clone_block,
        num_warps=4,
    )

    block_n = 32
    grid = (triton.cdiv(MATRIX_COLS, block_n), NUM_CHUNKS)
    _multi_source_partial_sum_kernel[grid](
        in0,
        in1,
        in2,
        in3,
        in4,
        in5,
        in6,
        in7,
        in8,
        in9,
        in10,
        clone_base,
        partials,
        MATRIX_COLS_=MATRIX_COLS,
        REDUCTION_CHUNK_=REDUCTION_CHUNK,
        NUM_CHUNKS_=NUM_CHUNKS,
        BLOCK_N=block_n,
        num_warps=4,
    )
    _finalize_sum_kernel[(triton.cdiv(MATRIX_COLS, block_n),)](
        partials,
        summed,
        MATRIX_COLS_=MATRIX_COLS,
        NUM_CHUNKS_=NUM_CHUNKS,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return (clone_base.as_strided((MATRIX_ROWS, MATRIX_COLS), (1, MATRIX_COLS)), summed)


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
