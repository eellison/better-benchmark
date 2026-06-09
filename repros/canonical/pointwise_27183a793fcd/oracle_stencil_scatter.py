"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT masked-LM slice/stencil/index_put scope by initializing the column-major transposed logits layout and atomically accumulating the three-tap token-neighbor stencil directly from mm_724 through arg0_1, whereas tuned Inductor reaches comparable time for the full scope; Inductor cannot materially improve this repro with a local layout/stencil fusion because the mandatory work is the 3.9M-element transposed logits write plus 4.19M indexed fp32 atomic updates into that layout; the fix is BANDWIDTH_BOUND: record this as at-floor unless a broader scatter algorithm changes the required atomic/update traffic."""
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


VOCAB = 30522
SEQ = 128
BATCH = 256
HIDDEN = 384
OUT0_ROWS = 384
MM_COLS = 30524
N_POS = BATCH * SEQ
OUT1_NUMEL = VOCAB * SEQ


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _init_transposed_logits_kernel(
        mm_ptr,
        out_ptr,
        N: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        MM_COLS_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vocab = offsets % VOCAB_SIZE
        col = offsets // VOCAB_SIZE
        vals = tl.load(mm_ptr + col * MM_COLS_SIZE + vocab, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, vals, mask=mask)

    @triton.jit
    def _scatter_stencil_kernel(
        mm_724_ptr,
        arg0_ptr,
        full_ptr,
        out_ptr,
        N_POS_SIZE: tl.constexpr,
        VOCAB_SIZE: tl.constexpr,
        SEQ_SIZE: tl.constexpr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pos = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

        valid_pos = pos < N_POS_SIZE
        valid_cols = cols < SEQ_SIZE
        q = pos % SEQ_SIZE
        idx = tl.load(arg0_ptr + pos, mask=valid_pos, other=0)

        pos_2d = pos[:, None]
        cols_2d = cols[None, :]
        valid = valid_pos[:, None] & valid_cols[None, :]

        center = tl.load(
            mm_724_ptr + pos_2d * HIDDEN_SIZE + (SEQ_SIZE + cols_2d),
            mask=valid,
            other=0.0,
        )
        right = tl.load(
            mm_724_ptr + (pos_2d + 1) * HIDDEN_SIZE + (2 * SEQ_SIZE + cols_2d),
            mask=valid & (q[:, None] < (SEQ_SIZE - 1)),
            other=0.0,
        )
        left = tl.load(
            mm_724_ptr + (pos_2d - 1) * HIDDEN_SIZE + cols_2d,
            mask=valid & (q[:, None] > 0),
            other=0.0,
        )

        value = (center + right) + left
        full_value = tl.load(full_ptr)
        value = tl.where(idx[:, None] == 0, full_value, value)

        out_offsets = idx[:, None] + cols_2d * VOCAB_SIZE
        tl.atomic_add(out_ptr + out_offsets, value, sem="relaxed", mask=valid)


def _validate_inputs(inputs):
    if len(inputs) != 5:
        raise ValueError(f"expected 5 inputs, got {len(inputs)}")
    mm, mm_724, arg0_1, full_1, shape_param = inputs
    if tuple(mm.shape) != (512, MM_COLS):
        raise ValueError(f"unexpected mm shape: {tuple(mm.shape)}")
    if tuple(mm_724.shape) != (N_POS, HIDDEN):
        raise ValueError(f"unexpected mm_724 shape: {tuple(mm_724.shape)}")
    if tuple(arg0_1.shape) != (BATCH, SEQ):
        raise ValueError(f"unexpected arg0_1 shape: {tuple(arg0_1.shape)}")
    if tuple(full_1.shape) != ():
        raise ValueError(f"unexpected full_1 shape: {tuple(full_1.shape)}")
    if list(shape_param) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter: {shape_param}")
    if mm.dtype != torch.float32 or mm_724.dtype != torch.float32 or full_1.dtype != torch.float32:
        raise ValueError("oracle_stencil_scatter.py expects f32 mm/mm_724/full_1")
    if arg0_1.dtype != torch.int64:
        raise ValueError(f"unexpected arg0_1 dtype: {arg0_1.dtype}")
    if not (mm.is_cuda and mm_724.is_cuda and arg0_1.is_cuda and full_1.is_cuda):
        raise ValueError("oracle_stencil_scatter.py expects CUDA tensor inputs")
    return mm, mm_724, arg0_1, full_1


@oracle_impl(hardware="H100", shapes="(T([512, 30524], f32), T([32768, 384], f32), T([256, 128], i64, gen=Index(30522)), T([], f32), S([256, 128, 384]))")
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
        raise RuntimeError("Triton is required for oracle_stencil_scatter.py")

    mm, mm_724, arg0_1, full_1 = _validate_inputs(inputs)

    slice_tensor_2 = mm[128:512, :VOCAB]
    add_tensor_2 = torch.empty_strided(
        (VOCAB, SEQ),
        (1, VOCAB),
        device=mm.device,
        dtype=torch.float32,
    )

    init_block = 1024
    _init_transposed_logits_kernel[(triton.cdiv(OUT1_NUMEL, init_block),)](
        mm,
        add_tensor_2,
        N=OUT1_NUMEL,
        VOCAB_SIZE=VOCAB,
        MM_COLS_SIZE=MM_COLS,
        BLOCK_N=init_block,
    )

    block_m = 1
    block_k = 128
    grid = (triton.cdiv(N_POS, block_m), triton.cdiv(SEQ, block_k))
    _scatter_stencil_kernel[grid](
        mm_724,
        arg0_1,
        full_1,
        add_tensor_2,
        N_POS_SIZE=N_POS,
        VOCAB_SIZE=VOCAB,
        SEQ_SIZE=SEQ,
        HIDDEN_SIZE=HIDDEN,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
    )

    return (slice_tensor_2, add_tensor_2)


def _check_layout(instance, inputs):
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)

    ok = True
    for index, (expected_i, actual_i) in enumerate(zip(expected, actual)):
        layout_ok = (
            expected_i.shape == actual_i.shape
            and expected_i.dtype == actual_i.dtype
            and expected_i.stride() == actual_i.stride()
            and expected_i.storage_offset() == actual_i.storage_offset()
        )
        if index == 0:
            layout_ok = layout_ok and (
                expected_i.untyped_storage().data_ptr()
                == actual_i.untyped_storage().data_ptr()
            )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_i.shape)} stride={actual_i.stride()} "
            f"storage_offset={actual_i.storage_offset()})"
        )
        ok = ok and layout_ok
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
