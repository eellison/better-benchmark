"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Llama embedding/layernorm-backward tuple by copying the incoming bf16 vocabulary gradient once, then directly reducing the 2048 token-row contributors into that final output while also accumulating the sibling hidden reduction, whereas Inductor materializes a full f32 zero tensor, scatters into it with `index_put(accumulate=True)`, casts it to bf16, and only then adds it to `mm`; Inductor cannot do this today because scheduler/codegen does not represent indexed embedding-gradient accumulation with surrounding rowwise layernorm math and sibling reductions as a structured scatter-reduce producer; the fix is SCATTER_REDUCE: add multi-source indexed scatter-reduce lowering that fuses the rowwise producer, `idx == -1` mask handling, sibling hidden reduction, and final `mm` add without the full f32 scatter intermediate."""
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

BATCH = 4
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 2048
VOCAB = 128256
INIT_BLOCK = 4096
BLOCK_H = 2048
FIX_BLOCK_H = 8
FIX_BLOCK_R = ROWS
INV_HIDDEN = 1.0 / HIDDEN
_DUPLICATE_TARGET_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _duplicate_targets(index: torch.Tensor) -> torch.Tensor:
    """Return wrapped target rows that receive more than one source row."""
    key = (index.data_ptr(), index.numel())
    cached = _DUPLICATE_TARGET_CACHE.get(key)
    if cached is not None:
        return cached

    counts: dict[int, int] = {}
    for raw in index.detach().cpu().reshape(-1).tolist():
        target = int(raw)
        if target < 0:
            target += VOCAB
        if 0 <= target < VOCAB:
            counts[target] = counts.get(target, 0) + 1
    targets = [target for target, count in counts.items() if count > 1]
    target_tensor = torch.tensor(targets, device=index.device, dtype=torch.int64)
    _DUPLICATE_TARGET_CACHE[key] = target_tensor
    return target_tensor

if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm_ptr,
        out1_ptr,
        out0_accum_ptr,
        TOTAL_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        copy_mask = offsets < TOTAL_
        values = tl.load(mm_ptr + offsets, mask=copy_mask, other=0.0)
        tl.store(out1_ptr + offsets, values, mask=copy_mask)

        zero_mask = offsets < HIDDEN_
        tl.store(out0_accum_ptr + offsets, tl.zeros([BLOCK_], tl.float32), mask=zero_mask)

    @triton.jit
    def _row_reduce_scatter_kernel(
        mm221_ptr,
        mm223_ptr,
        mm225_ptr,
        weight_ptr,
        x_ptr,
        scale_ptr,
        add203_ptr,
        index_ptr,
        full_ptr,
        out0_accum_ptr,
        src_rows_ptr,
        out1_ptr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
        INV_HIDDEN_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        mask = h < HIDDEN_
        base = row * HIDDEN_ + h

        mm221 = tl.load(mm221_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mm223 = tl.load(mm223_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mm225 = tl.load(mm225_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sum01 = (mm221 + mm223).to(tl.bfloat16).to(tl.float32)
        summed = (sum01 + mm225).to(tl.bfloat16).to(tl.float32)

        weight = tl.load(weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)

        scaled_x = (x * scale).to(tl.bfloat16).to(tl.float32)
        out0_contrib = (summed * scaled_x).to(tl.bfloat16).to(tl.float32)
        tl.atomic_add(out0_accum_ptr + h, out0_contrib, sem="relaxed", mask=mask)

        weighted = (summed * weight).to(tl.bfloat16).to(tl.float32)
        row_sum = tl.sum(tl.where(mask, weighted * x, 0.0), axis=0)
        scale_sq = scale * scale
        correction = (-0.5 * row_sum * scale_sq * scale) * INV_HIDDEN_ * (x * 2.0)
        add2 = weighted * scale + correction
        add2_bf = add2.to(tl.bfloat16).to(tl.float32)
        add203 = tl.load(add203_ptr + base, mask=mask, other=0.0).to(tl.float32)
        src = (add203 + add2_bf).to(tl.bfloat16).to(tl.float32)

        index = tl.load(index_ptr + row).to(tl.int64)
        full_value = tl.load(full_ptr).to(tl.float32)
        src = tl.where(index == -1, full_value, src)
        tl.store(src_rows_ptr + base, src, mask=mask)

        dest_index = tl.where(index < 0, index + VOCAB_, index)
        dest_valid = (dest_index >= 0) & (dest_index < VOCAB_)
        dest_offsets = dest_index * HIDDEN_ + h
        tl.atomic_add(out1_ptr + dest_offsets, src, sem="relaxed", mask=mask & dest_valid)

    @triton.jit
    def _duplicate_target_fix_kernel(
        src_rows_ptr,
        index_ptr,
        duplicate_targets_ptr,
        mm_ptr,
        out1_ptr,
        HIDDEN_: tl.constexpr,
        ROWS_: tl.constexpr,
        VOCAB_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        duplicate_id = tl.program_id(1)
        r = tl.arange(0, BLOCK_R_)
        h_mask = h < HIDDEN_
        r_mask = r < ROWS_

        target = tl.load(duplicate_targets_ptr + duplicate_id).to(tl.int64)
        raw_index = tl.load(index_ptr + r, mask=r_mask, other=0).to(tl.int64)
        dest_index = tl.where(raw_index < 0, raw_index + VOCAB_, raw_index)
        row_matches = r_mask & (dest_index == target)
        offsets = r[:, None] * HIDDEN_ + h[None, :]
        values = tl.load(
            src_rows_ptr + offsets,
            mask=row_matches[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        reduced = tl.sum(values, axis=0)
        scatter_bf = reduced.to(tl.bfloat16).to(tl.float32)
        target_offsets = target * HIDDEN_ + h
        mm_values = tl.load(mm_ptr + target_offsets, mask=h_mask, other=0.0).to(tl.float32)
        final = (mm_values + scatter_bf).to(tl.bfloat16)
        tl.store(out1_ptr + target_offsets, final, mask=h_mask)

    @triton.jit
    def _finalize_out0_kernel(
        out0_accum_ptr,
        out0_ptr,
        HIDDEN_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        h = tl.arange(0, BLOCK_H_)
        mask = h < HIDDEN_
        values = tl.load(out0_accum_ptr + h, mask=mask, other=0.0).to(tl.bfloat16)
        tl.store(out0_ptr + h, values, mask=mask)


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
        raise RuntimeError("Triton is required for this oracle")

    (
        mm_221,
        mm_223,
        mm_225,
        arg3_1,
        arg148_1,
        arg149_1,
        add_203,
        arg0_1,
        full_1,
        mm,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4

    assert mm_221.shape == (ROWS, HIDDEN)
    assert mm_223.shape == (ROWS, HIDDEN)
    assert mm_225.shape == (ROWS, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg148_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg149_1.shape == (BATCH, SEQ, 1)
    assert add_203.shape == (BATCH, SEQ, HIDDEN)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm.shape == (VOCAB, HIDDEN)

    out0_accum = torch.empty((HIDDEN,), device=mm.device, dtype=torch.float32)
    src_rows = torch.empty((ROWS, HIDDEN), device=mm.device, dtype=torch.float32)
    out0 = torch.empty((HIDDEN,), device=mm.device, dtype=torch.bfloat16)
    out1 = torch.empty_like(mm)
    duplicate_targets = _duplicate_targets(arg0_1)

    total = mm.numel()
    _init_outputs_kernel[(triton.cdiv(total, INIT_BLOCK),)](
        mm,
        out1,
        out0_accum,
        TOTAL_=total,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=8,
    )
    _row_reduce_scatter_kernel[(ROWS,)](
        mm_221,
        mm_223,
        mm_225,
        arg3_1,
        arg148_1,
        arg149_1,
        add_203,
        arg0_1,
        full_1,
        out0_accum,
        src_rows,
        out1,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        INV_HIDDEN_=INV_HIDDEN,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )
    if duplicate_targets.numel() > 0:
        _duplicate_target_fix_kernel[
            (triton.cdiv(HIDDEN, FIX_BLOCK_H), duplicate_targets.numel())
        ](
            src_rows,
            arg0_1,
            duplicate_targets,
            mm,
            out1,
            HIDDEN_=HIDDEN,
            ROWS_=ROWS,
            VOCAB_=VOCAB,
            BLOCK_R_=FIX_BLOCK_R,
            BLOCK_H_=FIX_BLOCK_H,
            num_warps=8,
        )
    _finalize_out0_kernel[(1,)](
        out0_accum,
        out0,
        HIDDEN_=HIDDEN,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )
    return (out0, out1)


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
