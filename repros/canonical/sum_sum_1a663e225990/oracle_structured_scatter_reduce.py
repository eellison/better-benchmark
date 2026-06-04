"""
Oracle for sum_sum_1a663e225990

Gap diagnosis:
  Classification: ALGEBRAIC_ELIMINATION
  What oracle does differently: Computes the full masked-LM softmax-backward update, both padded materialized outputs, and the sibling vocabulary reduction directly from labels/logits without constructing the dense one-hot tensor or reducing it.
  What Inductor change would fix: Add algebraic elimination for one-hot equality masks feeding a row reduction and dense softmax-backward epilogue, then fuse the remaining multi-output stores with the column reduction.
"""
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 16384
N = 30522
N_PAD = 30524
PAD = N_PAD - N
BLOCK_M = 128
BLOCK_N = 64
FINAL_BLOCK_C = 8

# Import shared oracle infrastructure (installed via pip install -e .)
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

if triton is not None:

    @triton.jit
    def _update_outputs_partial_kernel(
        numerator_ptr,
        denominator_ptr,
        label_ptr,
        logits_ptr,
        row_sub0_ptr,
        row_sub1_ptr,
        base_ptr,
        out0_ptr,
        out1_ptr,
        partial_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        N_PAD_: tl.constexpr,
        NUM_M_TILES_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
        rows = row_tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        cols = col_tile * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        row_mask = rows < M_
        col_mask = cols < N_
        mask = row_mask[:, None] & col_mask[None, :]

        labels = tl.load(label_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
        valid_label = (labels != -100) & (labels >= 0) & (labels < N_)
        scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)
        row_shift = (
            tl.load(row_sub0_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
            + tl.load(row_sub1_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        )

        offsets = rows[:, None] * N_ + cols[None, :]
        logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        base = tl.load(base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        exp_vals = tl.exp(logits - row_shift[:, None])
        update = scale * exp_vals
        values = base + tl.where(valid_label[:, None], update, 0.0)
        values = tl.where(mask, values, 0.0)

        out0_offsets = rows[:, None] * N_PAD_ + cols[None, :]
        out1_offsets = cols[None, :] * M_ + rows[:, None]
        tl.store(out0_ptr + out0_offsets, values, mask=mask)
        tl.store(out1_ptr + out1_offsets, values, mask=mask)

        partial = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + cols * NUM_M_TILES_ + row_tile,
            partial,
            mask=col_mask,
        )

    @triton.jit
    def _zero_padding_kernel(
        out0_ptr,
        out1_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        N_PAD_: tl.constexpr,
        PAD_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total0 = M_ * PAD_
        total = (M_ * PAD_) + (PAD_ * M_)
        active = offsets < total

        in_out0 = offsets < total0
        rem0 = offsets
        row0 = rem0 // PAD_
        pad_col = rem0 - row0 * PAD_
        tl.store(
            out0_ptr + row0 * N_PAD_ + (N_ + pad_col),
            0.0,
            mask=active & in_out0,
        )

        rem1 = offsets - total0
        pad_row = rem1 // M_
        col1 = rem1 - pad_row * M_
        tl.store(
            out1_ptr + (N_ + pad_row) * M_ + col1,
            0.0,
            mask=active & ~in_out0,
        )

    @triton.jit
    def _label_correction_kernel(
        numerator_ptr,
        denominator_ptr,
        label_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        N_PAD_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row_mask = rows < M_
        labels = tl.load(label_ptr + rows, mask=row_mask, other=-100).to(tl.int64)
        valid_label = row_mask & (labels != -100) & (labels >= 0) & (labels < N_)
        safe_labels = tl.where(valid_label, labels, 0)
        scale = tl.load(numerator_ptr).to(tl.float32) / tl.load(denominator_ptr).to(tl.float32)

        out0_offsets = rows * N_PAD_ + safe_labels
        out1_offsets = safe_labels * M_ + rows
        out0_vals = tl.load(out0_ptr + out0_offsets, mask=valid_label, other=0.0).to(tl.float32)
        out1_vals = tl.load(out1_ptr + out1_offsets, mask=valid_label, other=0.0).to(tl.float32)
        tl.store(out0_ptr + out0_offsets, out0_vals - scale, mask=valid_label)
        tl.store(out1_ptr + out1_offsets, out1_vals - scale, mask=valid_label)
        tl.atomic_add(out2_ptr + safe_labels, -scale, sem="relaxed", mask=valid_label)

    @triton.jit
    def _finalize_column_sum_kernel(
        partial_ptr,
        out2_ptr,
        N_: tl.constexpr,
        NUM_M_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tiles = tl.arange(0, BLOCK_TILES_)
        mask = (cols[:, None] < N_) & (tiles[None, :] < NUM_M_TILES_)
        values = tl.load(
            partial_ptr + cols[:, None] * NUM_M_TILES_ + tiles[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(values, axis=1)
        tl.store(out2_ptr + cols, sums, mask=cols < N_)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 13:
        raise ValueError(f"expected 13 inputs, got {len(inputs)}")

    (
        arg324_1,
        arg250_1,
        arg102_1,
        arg247_1,
        arg248_1,
        arg249_1,
        arg325_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    ) = inputs

    tensors = (arg324_1, arg250_1, arg102_1, arg247_1, arg248_1, arg249_1, arg325_1)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("oracle expects the repro tensor inputs in positions 0..6")
    if arg247_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if arg324_1.shape != () or arg250_1.shape != ():
        raise ValueError("expected scalar f32 scale inputs")
    if arg102_1.shape != (32, 512) or arg102_1.dtype != torch.int64:
        raise ValueError(f"unexpected label input: shape={tuple(arg102_1.shape)} dtype={arg102_1.dtype}")
    if arg247_1.shape != (32, 512, N) or arg325_1.shape != (32, 512, N):
        raise ValueError("unexpected logits/base tensor shape")
    if arg248_1.shape != (M, 1) or arg249_1.shape != (M, 1):
        raise ValueError("unexpected row subtraction tensor shape")
    if arg247_1.dtype != torch.float32 or arg325_1.dtype != torch.float32:
        raise ValueError("expected f32 logits/base tensors")
    if arg248_1.dtype != torch.float32 or arg249_1.dtype != torch.float32:
        raise ValueError("expected f32 row subtraction tensors")
    if not arg102_1.is_contiguous() or not arg247_1.is_contiguous() or not arg325_1.is_contiguous():
        raise ValueError("oracle expects contiguous captured inputs")

    return arg324_1, arg250_1, arg102_1, arg247_1, arg248_1, arg249_1, arg325_1


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
    arg324_1, arg250_1, arg102_1, arg247_1, arg248_1, arg249_1, arg325_1 = _validate_inputs(inputs)

    out0 = torch.empty_strided((M, N_PAD), (N_PAD, 1), device=arg247_1.device, dtype=torch.float32)
    out1 = torch.empty_strided((N_PAD, M), (M, 1), device=arg247_1.device, dtype=torch.float32)
    out2 = torch.empty_strided((N,), (1,), device=arg247_1.device, dtype=torch.float32)

    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    partial = torch.empty((N, num_m_tiles), device=arg247_1.device, dtype=torch.float32)

    _update_outputs_partial_kernel[(num_m_tiles, num_n_tiles)](
        arg324_1,
        arg250_1,
        arg102_1,
        arg247_1,
        arg248_1,
        arg249_1,
        arg325_1,
        out0,
        out1,
        partial,
        M_=M,
        N_=N,
        N_PAD_=N_PAD,
        NUM_M_TILES_=num_m_tiles,
        BLOCK_M_=BLOCK_M,
        BLOCK_N_=BLOCK_N,
        num_warps=8,
    )
    _zero_padding_kernel[(triton.cdiv((M * PAD) + (PAD * M), 256),)](
        out0,
        out1,
        M_=M,
        N_=N,
        N_PAD_=N_PAD,
        PAD_=PAD,
        BLOCK=256,
        num_warps=4,
    )
    _finalize_column_sum_kernel[(triton.cdiv(N, FINAL_BLOCK_C),)](
        partial,
        out2,
        N_=N,
        NUM_M_TILES_=num_m_tiles,
        BLOCK_C_=FINAL_BLOCK_C,
        BLOCK_TILES_=triton.next_power_of_2(num_m_tiles),
        num_warps=4,
    )
    _label_correction_kernel[(triton.cdiv(M, 256),)](
        arg324_1,
        arg250_1,
        arg102_1,
        out0,
        out1,
        out2,
        M_=M,
        N_=N,
        N_PAD_=N_PAD,
        BLOCK=256,
        num_warps=4,
    )
    return out0, out1, out2


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
