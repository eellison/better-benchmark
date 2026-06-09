"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Electra backward-like expression by recomputing the fixed hidden-size row formula inside Triton kernels and feeding its results directly into the two hidden reductions, the sequence and type-id accumulate scatters, and the vocabulary embedding-gradient accumulate scatter added to the incoming gradient tensor, whereas Inductor lowers the decomposed graph through separate reductions, where materializations, index_put accumulations, and a final add; Inductor cannot emit this today because its scheduler/codegen does not have a fused scatter-reduce template that can share a row reduction epilogue across multiple duplicate-index accumulate=True index_put users while preserving masked value semantics; the fix is SCATTER_REDUCE: add a gather/reduce/scatter-aware lowering that recognizes zero-initialized index_put accumulations fed by row-reduction epilogues and emits direct atomic/reduction kernels for all sibling scatter and reduction outputs."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 64
SEQ = 512
HIDDEN = 128
VOCAB = 30522
TYPE_VOCAB = 2
MASK_SCALE = 1.1111111111111112
INPUT_SHAPES = (
    (BATCH * SEQ, HIDDEN),
    (BATCH, SEQ, HIDDEN),
    (HIDDEN,),
    (BATCH, SEQ, HIDDEN),
    (BATCH, SEQ, 1),
    (1, SEQ),
    (),
    (1, SEQ),
    (BATCH, SEQ),
    (VOCAB, HIDDEN),
)
SMALL_OUTPUT_ELEMENTS = (SEQ + TYPE_VOCAB + 2) * HIDDEN
EMBED_OUTPUT_ELEMENTS = VOCAB * HIDDEN
INIT_BLOCK = 1024
TOKEN_BLOCK_S = 1
EMBED_BLOCK_M = 1

if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm_1_ptr,
        sum2_ptr,
        sum3_ptr,
        out512_ptr,
        out2_ptr,
        embed_out_ptr,
        embed_elements: tl.constexpr,
        hidden: tl.constexpr,
        seq: tl.constexpr,
        type_vocab: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        zeros = tl.zeros((block,), tl.float32)

        tl.store(
            embed_out_ptr + offsets,
            tl.load(mm_1_ptr + offsets, mask=offsets < embed_elements, other=0.0),
            mask=offsets < embed_elements,
        )

        sum2_start: tl.constexpr = 0
        sum3_start: tl.constexpr = hidden
        out512_start: tl.constexpr = 2 * hidden
        out2_start: tl.constexpr = (2 + seq) * hidden
        small_elements: tl.constexpr = (2 + seq + type_vocab) * hidden

        tl.store(sum2_ptr + offsets, zeros, mask=offsets < hidden)
        tl.store(
            sum3_ptr + offsets - sum3_start,
            zeros,
            mask=(offsets >= sum3_start) & (offsets < out512_start),
        )
        tl.store(
            out512_ptr + offsets - out512_start,
            zeros,
            mask=(offsets >= out512_start) & (offsets < out2_start),
        )
        tl.store(
            out2_ptr + offsets - out2_start,
            zeros,
            mask=(offsets >= out2_start) & (offsets < small_elements),
        )

    @triton.jit
    def _token_type_reduce_kernel(
        mm_148_ptr,
        bool_mask_ptr,
        gamma_ptr,
        arg104_ptr,
        arg325_ptr,
        arg1_ptr,
        full_ptr,
        arg103_ptr,
        sum2_ptr,
        sum3_ptr,
        out512_ptr,
        out2_ptr,
        batch: tl.constexpr,
        seq: tl.constexpr,
        hidden: tl.constexpr,
        type_vocab: tl.constexpr,
        mask_scale: tl.constexpr,
        block_s: tl.constexpr,
        block_h: tl.constexpr,
    ):
        s = tl.program_id(0) * block_s + tl.arange(0, block_s)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        flat_cols = tl.arange(0, block_h)
        gamma = tl.load(gamma_ptr + cols)
        full = tl.load(full_ptr)

        token_acc = tl.zeros((block_s, block_h), tl.float32)
        type_acc = tl.zeros((block_s, block_h), tl.float32)
        sum2_acc = tl.zeros((block_s, block_h), tl.float32)
        sum3_acc = tl.zeros((block_s, block_h), tl.float32)
        type_index = tl.load(arg103_ptr + s)

        for b in tl.range(0, batch):
            row = b * seq + s
            offsets = row * hidden + cols
            mm = tl.load(mm_148_ptr + offsets).to(tl.float32)
            keep = tl.load(bool_mask_ptr + offsets).to(tl.float32) * mask_scale
            arg104 = tl.load(arg104_ptr + offsets).to(tl.float32)
            row_scale = tl.load(arg325_ptr + row).to(tl.float32)

            mul1 = mm * keep
            p = mul1 * gamma
            sum_p = tl.sum(p, axis=1)[:, None]
            prod = p * arg104
            sum_prod = tl.sum(prod, axis=1)[:, None]
            g = row_scale * (p * hidden - sum_p - arg104 * sum_prod)

            token_acc += g
            type_acc += tl.where(type_index == -1, full, g)
            sum2_acc += mul1 * arg104
            sum3_acc += mul1

        token_index = tl.load(arg1_ptr + s)
        token_dst = tl.where(token_index < 0, token_index + seq, token_index)
        token_value = tl.where(token_index == -1, full, token_acc)
        tl.atomic_add(out512_ptr + token_dst * hidden + cols, token_value, sem="relaxed")

        type_dst = tl.where(type_index < 0, type_index + type_vocab, type_index)
        tl.atomic_add(out2_ptr + type_dst * hidden + cols, type_acc, sem="relaxed")

        tl.atomic_add(sum2_ptr + flat_cols, tl.sum(sum2_acc, axis=0), sem="relaxed")
        tl.atomic_add(sum3_ptr + flat_cols, tl.sum(sum3_acc, axis=0), sem="relaxed")

    @triton.jit
    def _embedding_scatter_kernel(
        mm_148_ptr,
        bool_mask_ptr,
        gamma_ptr,
        arg104_ptr,
        arg325_ptr,
        full_ptr,
        arg0_ptr,
        embed_out_ptr,
        seq: tl.constexpr,
        hidden: tl.constexpr,
        vocab: tl.constexpr,
        mask_scale: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0) * block_m + tl.arange(0, block_m)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        offsets = row * hidden + cols

        mm = tl.load(mm_148_ptr + offsets).to(tl.float32)
        keep = tl.load(bool_mask_ptr + offsets).to(tl.float32) * mask_scale
        gamma = tl.load(gamma_ptr + cols).to(tl.float32)
        arg104 = tl.load(arg104_ptr + offsets).to(tl.float32)
        row_scale = tl.load(arg325_ptr + row).to(tl.float32)

        mul1 = mm * keep
        p = mul1 * gamma
        sum_p = tl.sum(p, axis=1)[:, None]
        prod = p * arg104
        sum_prod = tl.sum(prod, axis=1)[:, None]
        g = row_scale * (p * hidden - sum_p - arg104 * sum_prod)

        raw_index = tl.load(arg0_ptr + row)
        dst = tl.where(raw_index < 0, raw_index + vocab, raw_index)
        full = tl.load(full_ptr)
        value = tl.where(raw_index == 0, full, g)
        tl.atomic_add(embed_out_ptr + dst * hidden + cols, value, sem="relaxed")


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        mm_148,
        arg105_1,
        arg3_1,
        arg104_1,
        arg325_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        mm_1,
        shape0,
        shape1,
    ) = inputs

    tensor_inputs = (
        mm_148,
        arg105_1,
        arg3_1,
        arg104_1,
        arg325_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        mm_1,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first ten repro inputs must be tensors")

    expected_dtypes = (
        torch.float32,
        torch.bool,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.int64,
        torch.float32,
        torch.int64,
        torch.int64,
        torch.float32,
    )
    for index, (value, expected_shape, expected_dtype) in enumerate(
        zip(tensor_inputs, INPUT_SHAPES, expected_dtypes)
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != expected_dtype:
            raise TypeError(f"input {index} dtype {value.dtype} != {expected_dtype}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (BATCH, SEQ):
        raise ValueError(f"unexpected expand shape parameter: {shape1!r}")

    return tensor_inputs


@oracle_impl(hardware="H100", shapes="(T([32768, 128], f32), T([64, 512, 128], b8), T([128], f32), T([64, 512, 128], f32), T([64, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([64, 512], i64, gen=Index(30522)), T([30522, 128], f32), S([64, 512, 128]), S([64, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
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
        mm_148,
        arg105_1,
        arg3_1,
        arg104_1,
        arg325_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        mm_1,
    ) = _validate_inputs(inputs)

    sum_dim_int_list_2 = torch.empty((HIDDEN,), device=mm_148.device, dtype=torch.float32)
    sum_dim_int_list_3 = torch.empty((HIDDEN,), device=mm_148.device, dtype=torch.float32)
    index_put_default = torch.empty((SEQ, HIDDEN), device=mm_148.device, dtype=torch.float32)
    index_put_default_1 = torch.empty((TYPE_VOCAB, HIDDEN), device=mm_148.device, dtype=torch.float32)
    add_tensor = torch.empty_like(mm_1)

    init_grid = (triton.cdiv(max(EMBED_OUTPUT_ELEMENTS, SMALL_OUTPUT_ELEMENTS), INIT_BLOCK),)
    _init_outputs_kernel[init_grid](
        mm_1,
        sum_dim_int_list_2,
        sum_dim_int_list_3,
        index_put_default,
        index_put_default_1,
        add_tensor,
        embed_elements=EMBED_OUTPUT_ELEMENTS,
        hidden=HIDDEN,
        seq=SEQ,
        type_vocab=TYPE_VOCAB,
        block=INIT_BLOCK,
        num_warps=4,
    )

    _token_type_reduce_kernel[(triton.cdiv(SEQ, TOKEN_BLOCK_S),)](
        mm_148,
        arg105_1,
        arg3_1,
        arg104_1,
        arg325_1,
        arg1_1,
        full_1,
        arg103_1,
        sum_dim_int_list_2,
        sum_dim_int_list_3,
        index_put_default,
        index_put_default_1,
        batch=BATCH,
        seq=SEQ,
        hidden=HIDDEN,
        type_vocab=TYPE_VOCAB,
        mask_scale=MASK_SCALE,
        block_s=TOKEN_BLOCK_S,
        block_h=HIDDEN,
        num_warps=4,
    )

    _embedding_scatter_kernel[(triton.cdiv(BATCH * SEQ, EMBED_BLOCK_M),)](
        mm_148,
        arg105_1,
        arg3_1,
        arg104_1,
        arg325_1,
        full_1,
        arg0_1,
        add_tensor,
        seq=SEQ,
        hidden=HIDDEN,
        vocab=VOCAB,
        mask_scale=MASK_SCALE,
        block_m=EMBED_BLOCK_M,
        block_h=HIDDEN,
        num_warps=4,
    )

    return (
        sum_dim_int_list_2,
        sum_dim_int_list_3,
        index_put_default,
        index_put_default_1,
        add_tensor,
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
