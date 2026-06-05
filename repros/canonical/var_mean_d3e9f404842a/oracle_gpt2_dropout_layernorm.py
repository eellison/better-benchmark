"""Gap diagnosis (classification: NEW_PATTERN): this diagnosis-only oracle structurally computes the complete GPT-2 training embedding, generated-position embedding, adjacent-position bool side output, Inductor-seeded random dropout, fp32 hidden-size-768 var_mean LayerNorm, transposed affine output, and rsqrt/768 side output with one shape-specialized Triton row kernel, whereas Inductor lowers the same decomposed embedding/iota/cat/slice/ne/random/dropout/var_mean/affine/view/permute graph through generic embedding, stochastic pointwise, normalization, and layout scheduling; Inductor cannot do this today because its pattern library does not canonicalize GPT-2 embedding plus generated-position dropout LayerNorm with sibling bool and invstd side outputs into one fixed-hidden stochastic embedding-layernorm template, and this artifact is not a verified floor because stochastic output values are skipped by the harness; the fix is NEW_PATTERN: add a GPT-2 training embedding-dropout-layernorm lowering that folds token and position gathers, dropout, row normalization, affine transpose epilogue, and side outputs into one specialized schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


N_BATCH = 4
SEQ_LEN = 512
HIDDEN = 768
VOCAB = 50257
POSITION_VOCAB = 1024
N_ROWS = N_BATCH * SEQ_LEN
N_SEEDS = 25
SEED_INDEX = 0
EPS = 1.0e-5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _gpt2_dropout_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        ne_out_ptr,
        out_base_ptr,
        invstd_div_out_ptr,
        seed_index: tl.constexpr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        position_id = row % seq_len

        token = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        random_offsets = (row * hidden + cols).to(tl.uint32)
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, random_offsets)

        embedded = token + position
        dropped = tl.where(random > dropout_p, embedded * dropout_scale, 0.0)
        dropped_for_reduce = tl.where(col_mask, dropped, 0.0)

        mean = tl.sum(dropped_for_reduce, axis=0) / hidden
        centered = dropped - mean
        centered_for_var = tl.where(col_mask, centered, 0.0)
        variance = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias

        tl.store(out_base_ptr + row * hidden + cols, out, mask=col_mask)
        tl.store(invstd_div_out_ptr + row, invstd / hidden)
        tl.store(ne_out_ptr + row, False)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    word_table, token_ids, position_table, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (word_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (N_BATCH, SEQ_LEN),
        (POSITION_VOCAB, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if word_table.dtype != torch.float32 or position_table.dtype != torch.float32:
        raise TypeError("embedding tables must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm weight and bias must be torch.float32")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if shape0_tuple != (N_BATCH, -1):
        raise ValueError(f"unexpected expand shape parameter: {shape0!r}")
    if shape1_tuple != (-1, HIDDEN):
        raise ValueError(f"unexpected view shape parameter: {shape1!r}")

    return word_table, token_ids, position_table, weight, bias, shape0_tuple, shape1_tuple


def oracle_gpt2_dropout_layernorm(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape1: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Repro.forward return for the fixed GPT-2 shape."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_gpt2_dropout_layernorm.py")

    seeds = torch.ops.prims.inductor_seeds.default(N_SEEDS, word_table.device)
    ne_out = torch.empty_strided(
        (N_BATCH, SEQ_LEN),
        (SEQ_LEN, 1),
        device=word_table.device,
        dtype=torch.bool,
    )
    out_base = torch.empty_strided(
        (N_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    invstd_div = torch.empty_strided(
        (N_BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=word_table.device,
        dtype=torch.float32,
    )

    _gpt2_dropout_layernorm_kernel[(N_ROWS,)](
        word_table,
        token_ids,
        position_table,
        weight,
        bias,
        seeds,
        ne_out,
        out_base,
        invstd_div,
        seed_index=SEED_INDEX,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        eps=EPS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        num_warps=1,
        num_stages=3,
    )
    return ne_out, out_base.view(shape1).permute(1, 0), invstd_div


def oracle_forward(inputs):
    """Run the full Repro.forward scope: mask, stochastic dropout LayerNorm, transpose, and invstd side output."""
    word_table, token_ids, position_table, weight, bias, _shape0, shape1 = _validate_inputs(inputs)
    return oracle_gpt2_dropout_layernorm(
        word_table,
        token_ids,
        position_table,
        weight,
        bias,
        shape1,
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
