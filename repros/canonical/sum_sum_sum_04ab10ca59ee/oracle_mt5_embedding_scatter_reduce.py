"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full MT5 dual RMSNorm/dropout backward tuple by copying the live vocabulary-gradient base once, computing each row producer once per branch, accumulating both hidden-column reductions, and directly adding valid indexed contributors under the sentinel mask into the dense output, whereas Inductor materializes two zero-filled index_put(accumulate=True) scatter intermediates and schedules the sibling reductions, pointwise producers, scatters, and dense adds as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured index_put scatter-reduce node that shares rowwise RMSNorm reductions across sibling hidden reductions and masked duplicate-index accumulation into an existing base tensor; the fix is SCATTER_REDUCE: add a gather-mask/scatter-reduce lowering for embedding-backward index_put(accumulate=True) that fuses the row producer, sentinel handling, column reductions, and direct base-update epilogue."""
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


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 512
VOCAB = 250112
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 4096
HIDDEN_BLOCK = 512


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
    if tensor.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous for this captured layout")


def _require_shape_param(name: str, value, expected: tuple[int, ...]) -> None:
    if tuple(value) != expected:
        raise ValueError(f"{name} is {value}, expected {list(expected)}")


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm_ptr,
        out0_ptr,
        out1_ptr,
        out_vocab_ptr,
        TOTAL_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        active = offsets < TOTAL_
        values = tl.load(mm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(out_vocab_ptr + offsets, values, mask=active)

        hidden_active = offsets < HIDDEN_
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=hidden_active)
        tl.store(out1_ptr + offsets, zeros, mask=hidden_active)

    @triton.jit
    def _branch_rmsnorm_scatter_reduce_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        weight_ptr,
        mask_ptr,
        saved_ptr,
        rstd_ptr,
        add_ptr,
        index_ptr,
        full_ptr,
        out_reduce_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        hidden_mask = h < HIDDEN_
        base = row * HIDDEN_ + h

        added = tl.load(mm0_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        added += tl.load(mm1_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        added += tl.load(mm2_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)

        keep = tl.load(mask_ptr + base, mask=hidden_mask, other=0).to(tl.float32)
        saved = tl.load(saved_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        weight = tl.load(weight_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        prior_grad = tl.load(add_ptr + base, mask=hidden_mask, other=0.0).to(tl.float32)

        dropped_saved = keep * saved * DROPOUT_SCALE_
        weighted = added * weight
        row_dot = tl.sum(tl.where(hidden_mask, weighted * dropped_saved, 0.0), axis=0)

        reduce_contrib = added * dropped_saved * rstd
        tl.atomic_add(out_reduce_ptr + h, reduce_contrib, sem="relaxed", mask=hidden_mask)

        correction = row_dot * rstd * rstd * rstd * dropped_saved * (-1.0 / HIDDEN_)
        grad = (prior_grad + weighted * rstd + correction) * keep * DROPOUT_SCALE_

        raw_index = tl.load(index_ptr + row).to(tl.int64)
        wrapped_index = tl.where(raw_index < 0, raw_index + VOCAB_, raw_index)
        full_value = tl.load(full_ptr).to(tl.float32)
        scatter_value = tl.where(raw_index == -1, full_value, grad)
        valid_index = (wrapped_index >= 0) & (wrapped_index < VOCAB_)
        tl.atomic_add(
            out_vocab_ptr + wrapped_index * HIDDEN_ + h,
            scatter_value,
            sem="relaxed",
            mask=hidden_mask & valid_index,
        )


def oracle_mt5_embedding_scatter_reduce(
    mm_173: torch.Tensor,
    mm_175: torch.Tensor,
    mm_177: torch.Tensor,
    arg76_1: torch.Tensor,
    arg319_1: torch.Tensor,
    arg189_1: torch.Tensor,
    arg320_1: torch.Tensor,
    add_123: torch.Tensor,
    arg0_1: torch.Tensor,
    full_1: torch.Tensor,
    mm: torch.Tensor,
    mm_285: torch.Tensor,
    mm_287: torch.Tensor,
    mm_289: torch.Tensor,
    arg2_1: torch.Tensor,
    arg191_1: torch.Tensor,
    arg192_1: torch.Tensor,
    add_220: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    row_shape = (BATCH, SEQ, HIDDEN)
    hidden_shape = (HIDDEN,)
    _require_shape_param("_shape_param_0", _shape_param_0, row_shape)
    _require_shape_param("_shape_param_1", _shape_param_1, row_shape)
    _require_shape_param("_shape_param_2", _shape_param_2, row_shape)
    _require_shape_param("_shape_param_3", _shape_param_3, hidden_shape)
    _require_shape_param("_shape_param_4", _shape_param_4, row_shape)
    _require_shape_param("_shape_param_5", _shape_param_5, row_shape)
    _require_shape_param("_shape_param_6", _shape_param_6, row_shape)
    _require_shape_param("_shape_param_7", _shape_param_7, row_shape)
    _require_shape_param("_shape_param_8", _shape_param_8, hidden_shape)
    _require_shape_param("_shape_param_9", _shape_param_9, row_shape)

    flat_shape = (ROWS, HIDDEN)
    rstd_shape = (BATCH, SEQ, 1)
    index_shape = (BATCH, SEQ)
    vocab_shape = (VOCAB, HIDDEN)
    _require_tensor("mm_173", mm_173, flat_shape, torch.float32)
    _require_tensor("mm_175", mm_175, flat_shape, torch.float32)
    _require_tensor("mm_177", mm_177, flat_shape, torch.float32)
    _require_tensor("arg76_1", arg76_1, hidden_shape, torch.float32)
    _require_tensor("arg319_1", arg319_1, row_shape, torch.bool)
    _require_tensor("arg189_1", arg189_1, row_shape, torch.float32)
    _require_tensor("arg320_1", arg320_1, rstd_shape, torch.float32)
    _require_tensor("add_123", add_123, row_shape, torch.float32)
    _require_tensor("arg0_1", arg0_1, index_shape, torch.int64)
    _require_tensor("full_1", full_1, (), torch.float32)
    _require_tensor("mm", mm, vocab_shape, torch.float32)
    _require_tensor("mm_285", mm_285, flat_shape, torch.float32)
    _require_tensor("mm_287", mm_287, flat_shape, torch.float32)
    _require_tensor("mm_289", mm_289, flat_shape, torch.float32)
    _require_tensor("arg2_1", arg2_1, hidden_shape, torch.float32)
    _require_tensor("arg191_1", arg191_1, row_shape, torch.bool)
    _require_tensor("arg192_1", arg192_1, rstd_shape, torch.float32)
    _require_tensor("add_220", add_220, row_shape, torch.float32)

    out0 = torch.empty((HIDDEN,), device=mm.device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=mm.device, dtype=torch.float32)
    out_vocab = torch.empty_strided(
        tuple(mm.shape),
        tuple(mm.stride()),
        device=mm.device,
        dtype=mm.dtype,
    )

    _init_outputs_kernel[(triton.cdiv(VOCAB * HIDDEN, INIT_BLOCK),)](
        mm,
        out0,
        out1,
        out_vocab,
        TOTAL_=VOCAB * HIDDEN,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=4,
    )

    _branch_rmsnorm_scatter_reduce_kernel[(ROWS,)](
        mm_173,
        mm_175,
        mm_177,
        arg76_1,
        arg319_1,
        arg189_1,
        arg320_1,
        add_123,
        arg0_1,
        full_1,
        out0,
        out_vocab,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )

    _branch_rmsnorm_scatter_reduce_kernel[(ROWS,)](
        mm_285,
        mm_287,
        mm_289,
        arg2_1,
        arg191_1,
        arg189_1,
        arg192_1,
        add_220,
        arg0_1,
        full_1,
        out1,
        out_vocab,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )

    return out0, out1, out_vocab


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
    return oracle_mt5_embedding_scatter_reduce(*inputs)


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
