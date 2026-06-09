"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Qwen RMSNorm-backward weight-gradient and base-plus-indexed vocabulary-gradient output by sharing row reductions, copying the live bf16 base, and reducing duplicate indexed rows in a compact buffer instead of materializing the full f32 scatter intermediate, whereas Inductor currently builds the rowwise producer, full zero scatter buffer, `index_put(accumulate=True)`, cast, and base add as separate generic work; Inductor cannot do this today because scheduler/codegen lacks a gather-mask-reduce scatter-reduce lowering that keeps RMSNorm row summaries live while emitting the dense base-add output; the fix is SCATTER_REDUCE: add an indexed embedding-gradient lowering that folds sentinel masking, duplicate-row reduction, final bf16 cast, and base add into direct output codegen."""
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


BATCH = 4
SEQ = 512
TOKENS = BATCH * SEQ
HIDDEN = 1024
OUT_ROWS = 151936
MAX_DUP_ROWS = TOKENS // 2
BLOCK_OUT0_COLS = 8
BLOCK_OUT1_COLS = 512
BLOCK_COPY = 1024
BLOCK_FILL = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(shape_param: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape_param)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _bf16_add3(lhs, mid, rhs):
        first = (lhs + mid).to(tl.bfloat16).to(tl.float32)
        return (first + rhs).to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def _fill_i32_kernel(
        dst,
        value: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst + offsets, value, mask=offsets < n_elements)

    @triton.jit
    def _zero_f32_kernel(
        dst,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            dst + offsets,
            tl.zeros((BLOCK,), tl.float32),
            mask=offsets < n_elements,
        )

    @triton.jit
    def _copy_bf16_kernel(
        src,
        dst,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(src + offsets, mask=offsets < n_elements, other=0.0)
        tl.store(dst + offsets, values, mask=offsets < n_elements)

    @triton.jit
    def _row_dot_kernel(
        mm0,
        mm1,
        mm2,
        gamma,
        norm_input,
        row_dot,
        HIDDEN_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        token = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        offsets = token * HIDDEN_ + cols

        upstream = _bf16_add3(
            tl.load(mm0 + offsets).to(tl.float32),
            tl.load(mm1 + offsets).to(tl.float32),
            tl.load(mm2 + offsets).to(tl.float32),
        )
        weighted = (
            upstream * tl.load(gamma + cols).to(tl.float32)
        ).to(tl.bfloat16).to(tl.float32)
        x = tl.load(norm_input + offsets).to(tl.float32)
        tl.store(row_dot + token, tl.sum(weighted * x, axis=0))

    @triton.jit
    def _weight_grad_kernel(
        mm0,
        mm1,
        mm2,
        norm_input,
        inv_rms,
        out0,
        HIDDEN_: tl.constexpr,
        TOKENS_: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_R)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        mask = (rows[:, None] < TOKENS_) & col_mask[None, :]

        upstream = _bf16_add3(
            tl.load(mm0 + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(mm1 + offsets, mask=mask, other=0.0).to(tl.float32),
            tl.load(mm2 + offsets, mask=mask, other=0.0).to(tl.float32),
        )
        norm = (
            tl.load(norm_input + offsets, mask=mask, other=0.0).to(tl.float32)
            * tl.load(inv_rms + rows, mask=rows < TOKENS_, other=0.0).to(tl.float32)[:, None]
        ).to(tl.bfloat16).to(tl.float32)
        product = (upstream * norm).to(tl.bfloat16).to(tl.float32)
        tl.store(
            out0 + cols,
            tl.sum(tl.where(mask, product, 0.0), axis=0).to(tl.bfloat16),
            mask=col_mask,
        )

    @triton.jit
    def _count_index_rows_kernel(
        indices,
        counts,
        OUT_ROWS_: tl.constexpr,
    ):
        src = tl.program_id(0)
        raw_dest = tl.load(indices + src).to(tl.int64)
        dest = tl.where(raw_dest < 0, raw_dest + OUT_ROWS_, raw_dest)
        valid = (dest >= 0) & (dest < OUT_ROWS_)
        tl.atomic_add(counts + dest, 1, sem="relaxed", mask=valid)

    @triton.jit
    def _build_duplicate_rows_kernel(
        counts,
        row_to_dup_slot,
        dup_rows,
        dup_counter,
        OUT_ROWS_: tl.constexpr,
        MAX_DUP_ROWS_: tl.constexpr,
    ):
        row = tl.program_id(0)
        count = tl.load(counts + row).to(tl.int32)
        is_duplicate = count > 1
        slot = tl.atomic_add(dup_counter, 1, sem="relaxed", mask=is_duplicate)
        in_bounds = is_duplicate & (slot < MAX_DUP_ROWS_)
        tl.store(row_to_dup_slot + row, slot, mask=in_bounds)
        tl.store(dup_rows + slot, row, mask=in_bounds)

    @triton.jit
    def _indexed_update_or_accumulate_kernel(
        mm0,
        mm1,
        mm2,
        gamma,
        norm_input,
        inv_rms,
        add_tail,
        indices,
        full_value,
        base,
        row_dot,
        counts,
        row_to_dup_slot,
        out1,
        dup_accum,
        HIDDEN_: tl.constexpr,
        OUT_ROWS_: tl.constexpr,
        MAX_DUP_ROWS_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        src = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        raw_dest = tl.load(indices + src).to(tl.int64)
        dest = tl.where(raw_dest < 0, raw_dest + OUT_ROWS_, raw_dest)
        valid_dest = (dest >= 0) & (dest < OUT_ROWS_)
        active = valid_dest & col_mask
        offsets = src * HIDDEN_ + cols

        upstream = _bf16_add3(
            tl.load(mm0 + offsets, mask=active, other=0.0).to(tl.float32),
            tl.load(mm1 + offsets, mask=active, other=0.0).to(tl.float32),
            tl.load(mm2 + offsets, mask=active, other=0.0).to(tl.float32),
        )
        gamma_vals = tl.load(gamma + cols, mask=col_mask, other=0.0).to(tl.float32)
        weighted = (upstream * gamma_vals).to(tl.bfloat16).to(tl.float32)
        x = tl.load(norm_input + offsets, mask=active, other=0.0).to(tl.float32)
        inv = tl.load(inv_rms + src).to(tl.float32)
        dot = tl.load(row_dot + src).to(tl.float32)
        inv3 = (inv * inv) * inv
        correction = ((dot * -0.5) * inv3 / HIDDEN_) * (x * 2.0)
        input_grad = weighted * inv + correction
        input_grad_bf16 = input_grad.to(tl.bfloat16).to(tl.float32)
        add_vals = tl.load(add_tail + offsets, mask=active, other=0.0).to(tl.float32)
        token_grad = (add_vals + input_grad_bf16).to(tl.bfloat16).to(tl.float32)
        source_value = tl.where(raw_dest == -1, tl.load(full_value).to(tl.float32), token_grad)

        count = tl.load(counts + dest, mask=valid_dest, other=0).to(tl.int32)
        out_offsets = dest * HIDDEN_ + cols
        is_unique = active & (count == 1)
        base_vals = tl.load(base + out_offsets, mask=is_unique, other=0.0).to(tl.float32)
        scatter_vals = source_value.to(tl.bfloat16).to(tl.float32)
        tl.store(
            out1 + out_offsets,
            (base_vals + scatter_vals).to(tl.bfloat16),
            mask=is_unique,
        )

        slot = tl.load(row_to_dup_slot + dest, mask=valid_dest, other=-1).to(tl.int32)
        is_duplicate = active & (count > 1) & (slot >= 0) & (slot < MAX_DUP_ROWS_)
        tl.atomic_add(
            dup_accum + slot * HIDDEN_ + cols,
            source_value,
            sem="relaxed",
            mask=is_duplicate,
        )

    @triton.jit
    def _finalize_duplicate_rows_kernel(
        dup_accum,
        dup_rows,
        base,
        out1,
        HIDDEN_: tl.constexpr,
        MAX_DUP_ROWS_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        slot = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        col_mask = cols < HIDDEN_
        row = tl.load(dup_rows + slot, mask=slot < MAX_DUP_ROWS_, other=-1).to(tl.int32)
        active = (row >= 0) & col_mask
        offsets = row * HIDDEN_ + cols
        accum = tl.load(
            dup_accum + slot * HIDDEN_ + cols,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        scatter_vals = accum.to(tl.bfloat16).to(tl.float32)
        base_vals = tl.load(base + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(out1 + offsets, (base_vals + scatter_vals).to(tl.bfloat16), mask=active)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 15:
        raise ValueError(f"expected 15 Repro.forward inputs, got {len(inputs)}")

    (
        mm_389,
        mm_391,
        mm_393,
        arg3_1,
        arg312_1,
        arg313_1,
        add_413,
        arg0_1,
        full_1,
        mm,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    expected = {
        "mm_389": (mm_389, (TOKENS, HIDDEN), torch.bfloat16),
        "mm_391": (mm_391, (TOKENS, HIDDEN), torch.bfloat16),
        "mm_393": (mm_393, (TOKENS, HIDDEN), torch.bfloat16),
        "arg3_1": (arg3_1, (HIDDEN,), torch.bfloat16),
        "arg312_1": (arg312_1, (BATCH, SEQ, HIDDEN), torch.bfloat16),
        "arg313_1": (arg313_1, (BATCH, SEQ, 1), torch.float32),
        "add_413": (add_413, (BATCH, SEQ, HIDDEN), torch.bfloat16),
        "arg0_1": (arg0_1, (BATCH, SEQ), torch.int64),
        "full_1": (full_1, (), torch.float32),
        "mm": (mm, (OUT_ROWS, HIDDEN), torch.bfloat16),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected {dtype} shape={list(shape)}, "
                f"got dtype={tensor.dtype} shape={list(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the canonical oracle")

    if _shape_tuple(shape0) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected _shape_param_0: {shape0}")
    if _shape_tuple(shape1) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected _shape_param_1: {shape1}")
    if _shape_tuple(shape2) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected _shape_param_2: {shape2}")
    if _shape_tuple(shape3) != (HIDDEN,):
        raise ValueError(f"unexpected _shape_param_3: {shape3}")
    if _shape_tuple(shape4) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected _shape_param_4: {shape4}")


@oracle_impl(hardware="H100", shapes="(T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([4, 512], i64, gen=Index(151936)), T([], f32), T([151936, 1024], bf16), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512, 1024]), S([1024]), S([4, 512, 1024]))")
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
    _validate_inputs(inputs)
    (
        mm_389,
        mm_391,
        mm_393,
        arg3_1,
        arg312_1,
        arg313_1,
        add_413,
        arg0_1,
        full_1,
        mm,
        _shape0,
        _shape1,
        _shape2,
        shape3,
        _shape4,
    ) = inputs

    out0 = torch.empty(_shape_tuple(shape3), device=mm.device, dtype=torch.bfloat16)
    out1 = torch.empty((OUT_ROWS, HIDDEN), device=mm.device, dtype=torch.bfloat16)
    row_dot = torch.empty((TOKENS,), device=mm.device, dtype=torch.float32)
    counts = torch.empty((OUT_ROWS,), device=mm.device, dtype=torch.int32)
    row_to_dup_slot = torch.empty((OUT_ROWS,), device=mm.device, dtype=torch.int32)
    dup_counter = torch.empty((1,), device=mm.device, dtype=torch.int32)
    dup_rows = torch.empty((MAX_DUP_ROWS,), device=mm.device, dtype=torch.int32)
    dup_accum = torch.empty((MAX_DUP_ROWS, HIDDEN), device=mm.device, dtype=torch.float32)

    _copy_bf16_kernel[(triton.cdiv(OUT_ROWS * HIDDEN, BLOCK_COPY),)](
        mm,
        out1,
        n_elements=OUT_ROWS * HIDDEN,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _row_dot_kernel[(TOKENS,)](
        mm_389,
        mm_391,
        mm_393,
        arg3_1,
        arg312_1,
        row_dot,
        HIDDEN_=HIDDEN,
        BLOCK_C=HIDDEN,
        num_warps=4,
    )
    _weight_grad_kernel[(triton.cdiv(HIDDEN, BLOCK_OUT0_COLS),)](
        mm_389,
        mm_391,
        mm_393,
        arg312_1,
        arg313_1,
        out0,
        HIDDEN_=HIDDEN,
        TOKENS_=TOKENS,
        BLOCK_R=TOKENS,
        BLOCK_C=BLOCK_OUT0_COLS,
        num_warps=8,
    )
    _fill_i32_kernel[(triton.cdiv(OUT_ROWS, BLOCK_FILL),)](
        counts,
        value=0,
        n_elements=OUT_ROWS,
        BLOCK=BLOCK_FILL,
        num_warps=4,
    )
    _count_index_rows_kernel[(TOKENS,)](
        arg0_1,
        counts,
        OUT_ROWS_=OUT_ROWS,
        num_warps=4,
    )
    _fill_i32_kernel[(triton.cdiv(OUT_ROWS, BLOCK_FILL),)](
        row_to_dup_slot,
        value=-1,
        n_elements=OUT_ROWS,
        BLOCK=BLOCK_FILL,
        num_warps=4,
    )
    _fill_i32_kernel[(1,)](
        dup_counter,
        value=0,
        n_elements=1,
        BLOCK=1,
        num_warps=1,
    )
    _fill_i32_kernel[(triton.cdiv(MAX_DUP_ROWS, BLOCK_FILL),)](
        dup_rows,
        value=-1,
        n_elements=MAX_DUP_ROWS,
        BLOCK=BLOCK_FILL,
        num_warps=4,
    )
    _build_duplicate_rows_kernel[(OUT_ROWS,)](
        counts,
        row_to_dup_slot,
        dup_rows,
        dup_counter,
        OUT_ROWS_=OUT_ROWS,
        MAX_DUP_ROWS_=MAX_DUP_ROWS,
        num_warps=4,
    )
    _zero_f32_kernel[(triton.cdiv(MAX_DUP_ROWS * HIDDEN, BLOCK_COPY),)](
        dup_accum,
        n_elements=MAX_DUP_ROWS * HIDDEN,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )
    _indexed_update_or_accumulate_kernel[
        (TOKENS, triton.cdiv(HIDDEN, BLOCK_OUT1_COLS))
    ](
        mm_389,
        mm_391,
        mm_393,
        arg3_1,
        arg312_1,
        arg313_1,
        add_413,
        arg0_1,
        full_1,
        mm,
        row_dot,
        counts,
        row_to_dup_slot,
        out1,
        dup_accum,
        HIDDEN_=HIDDEN,
        OUT_ROWS_=OUT_ROWS,
        MAX_DUP_ROWS_=MAX_DUP_ROWS,
        BLOCK_C=BLOCK_OUT1_COLS,
        num_warps=4,
    )
    _finalize_duplicate_rows_kernel[
        (MAX_DUP_ROWS, triton.cdiv(HIDDEN, BLOCK_OUT1_COLS))
    ](
        dup_accum,
        dup_rows,
        mm,
        out1,
        HIDDEN_=HIDDEN,
        MAX_DUP_ROWS_=MAX_DUP_ROWS,
        BLOCK_C=BLOCK_OUT1_COLS,
        num_warps=4,
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
