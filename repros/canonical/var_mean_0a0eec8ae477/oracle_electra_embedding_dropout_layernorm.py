"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Electra embedding-layernorm-dropout return from Repro.forward in one fixed-hidden Triton row-block kernel, including gathered token-type ids, word/token-type/position embedding gathers, fp32 hidden-size-128 var_mean normalization, affine, Inductor-style dropout, final [32768,128] view, and the sibling rsqrt/128 side output, whereas Inductor lowers the same graph through generic embedding gathers, normalization scheduling, stochastic pointwise work, and side-output storage; Inductor cannot do this today because norm-template canonicalization does not recognize gathered embedding assembly plus stochastic dropout plus a live inverse-std output as one fixed-hidden semantic pattern; the fix is NEW_PATTERN: add an embedding-layernorm-dropout lowering that folds the indexed producers, row reduction, affine/dropout epilogue, and inverse-std side output into specialized codegen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops

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
ROWS = 64 * 512
BATCH = 64
SEQ_LEN = 512
HIDDEN = 128
WORD_VOCAB = 30522
TOKEN_TYPES = 2
EPS = 1.0e-12
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SEED_COUNT = 37
BLOCK_M = 8
BLOCK_H = 128

if triton is not None:

    @triton.jit
    def _electra_embedding_layernorm_dropout_kernel(
        token_type_ids_ptr,
        position_ids_ptr,
        word_embedding_ptr,
        word_ids_ptr,
        token_type_embedding_ptr,
        position_embedding_ptr,
        weight_ptr,
        bias_ptr,
        seed_ptr,
        dropped_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        rows_total: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        row_mask = rows < rows_total
        col_mask = cols < hidden
        elem_mask = row_mask[:, None] & col_mask[None, :]

        seq = rows % seq_len
        word_ids = tl.load(word_ids_ptr + rows, mask=row_mask, other=0)
        position_ids = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)
        token_type_ids = tl.load(token_type_ids_ptr + position_ids, mask=row_mask, other=0)

        word = tl.load(
            word_embedding_ptr + word_ids[:, None] * hidden + cols[None, :],
            mask=elem_mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_embedding_ptr + token_type_ids[:, None] * hidden + cols[None, :],
            mask=elem_mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + position_ids[:, None] * hidden + cols[None, :],
            mask=elem_mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(elem_mask, word + token_type + position, 0.0)
        mean = tl.sum(x, axis=1) / hidden
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(elem_mask, centered * centered, 0.0), axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        flat_offsets = rows[:, None] * hidden + cols[None, :]
        seed = tl.load(seed_ptr)
        random = tl.rand(seed, flat_offsets.to(tl.uint32))
        dropped = tl.where(random > dropout_p, y, 0.0) * dropout_scale

        tl.store(dropped_ptr + flat_offsets, dropped, mask=elem_mask)
        tl.store(invstd_div_ptr + rows, invstd / hidden, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_expand_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first eight repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (TOKEN_TYPES, HIDDEN),
        (SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")

    if not all(value.dtype == torch.int64 for value in (token_type_ids, position_ids, word_ids)):
        raise TypeError("token type ids, position ids, and word ids must be torch.int64")
    fp32_tensors = (word_embedding, token_type_embedding, position_embedding, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float32")

    token_type_expand_shape_tuple = _shape_tuple(token_type_expand_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if token_type_expand_shape_tuple != (BATCH, SEQ_LEN):
        raise ValueError(f"unexpected token-type expand shape: {token_type_expand_shape!r}")
    if output_shape_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected output view shape: {output_shape!r}")

    return (
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        token_type_expand_shape_tuple,
        output_shape_tuple,
    )


def _make_inductor_seed(device: torch.device) -> torch.Tensor:
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device=device)
    return torch.ops.prims.inductor_lookup_seed.default(seeds, 0)


@oracle_impl(hardware="H100", shapes="(T([1, 512], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), T([30522, 128], f32), T([64, 512], i64, gen=Index(30522)), T([2, 128], f32), T([512, 128], f32), T([128], f32), T([128], f32), S([64, 512]), S([32768, 128]))")
def oracle_forward(inputs):
    """Run the complete Electra embedding, LayerNorm, dropout, and side output scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_electra_embedding_dropout_layernorm.py")

    (
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        _token_type_expand_shape,
        output_shape,
    ) = _validate_inputs(inputs)

    dropped = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_embedding.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=word_embedding.device,
        dtype=torch.float32,
    )
    seed = _make_inductor_seed(word_embedding.device)

    _electra_embedding_layernorm_dropout_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        token_type_ids,
        position_ids,
        word_embedding,
        word_ids,
        token_type_embedding,
        position_embedding,
        weight,
        bias,
        seed,
        dropped,
        invstd_div,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        rows_total=ROWS,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return (torch.ops.aten.view.default(dropped, output_shape), invstd_div)


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
