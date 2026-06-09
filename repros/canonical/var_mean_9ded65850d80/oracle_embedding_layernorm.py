"""
Oracle for var_mean_9ded65850d80

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ALBERT embedding assembly and layernorm as a dedicated one-warp row Triton kernel, including the token embedding, gathered token-type embedding, position embedding, fp16 embedding-add rounding, fp32 variance/mean over hidden size 128, affine epilogue, fp16 cast, and final [512, 128] view, whereas Inductor lowers the same scope through its generic fused var_mean/norm reduction machinery with dynamic masks/asserts and generic reduction bookkeeping; Inductor cannot emit this exact floor today because norm-template canonicalization does not recognize embedding+gather producers as a semantic embedding-layernorm pattern with fixed hidden size and direct indexed loads; the fix is NEW_PATTERN: add an embedding-layernorm template that folds token/type/position indexed loads into a fixed-K row layernorm kernel and autotunes the one-warp shape.
"""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


SEQ_LEN = 512
HIDDEN = 128
VOCAB = 30000
TYPE_VOCAB = 2
EPS = 1.0e-12
DEFAULT_NUM_WARPS = 1
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        token_type_source_ptr,
        position_ids_ptr,
        token_type_table_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        eps: tl.constexpr,
        hidden: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, hidden)

        token_id = tl.load(token_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + row)
        token_type_id = tl.load(token_type_source_ptr + position_id)

        word = tl.load(word_table_ptr + token_id * hidden + cols)
        token_type = tl.load(token_type_table_ptr + token_type_id * hidden + cols)
        position = tl.load(position_table_ptr + position_id * hidden + cols)

        # The captured graph materializes fp16 after each embedding add before
        # promoting to fp32 for var_mean/layernorm.
        pair_sum = (word + token_type).to(tl.float16)
        summed = (pair_sum + position).to(tl.float16).to(tl.float32)

        mean = tl.sum(summed, axis=0) / hidden
        centered = summed - mean
        var = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(out_ptr + row * hidden + cols, out.to(tl.float16))


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if len(inputs) != 10:
        raise ValueError(f"expected 10 inputs, got {len(inputs)}")

    (
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        shape0,
        shape1,
    ) = inputs

    tensor_inputs = (
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first eight repro inputs must be tensors")

    expected_shapes = (
        (VOCAB, HIDDEN),
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (1, SEQ_LEN),
        (TYPE_VOCAB, HIDDEN),
        (SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    fp16_tensors = (word_table, token_type_table, position_table, weight, bias)
    if not all(value.dtype == torch.float16 for value in fp16_tensors):
        raise TypeError("embedding tables, layernorm weight, and bias must be torch.float16")
    index_tensors = (token_ids, token_type_source, position_ids)
    if not all(value.dtype == torch.int64 for value in index_tensors):
        raise TypeError("token, token-type source, and position ids must be torch.int64")

    if list(shape0) != [1, SEQ_LEN] or list(shape1) != [SEQ_LEN, HIDDEN]:
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return (
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
    )


def oracle_embedding_layernorm(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    token_type_source: torch.Tensor,
    position_ids: torch.Tensor,
    token_type_table: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Compute the full Repro.forward output with one Triton row kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    out = torch.empty_strided(
        (SEQ_LEN, HIDDEN),
        (HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float16,
    )
    _embedding_layernorm_kernel[(SEQ_LEN,)](
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        out,
        eps=EPS,
        hidden=HIDDEN,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([30000, 128], f16), T([1, 512], i64, gen=Index(30000)), T([1, 512], i64, gen=Index(2)), T([1, 512], i64, gen=Index(512)), T([2, 128], f16), T([512, 128], f16), T([128], f16), T([128], f16), S([1, 512]), S([512, 128]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single [512, 128] float16 output tensor.
    """
    tensor_inputs = _validate_inputs(inputs)
    return oracle_embedding_layernorm(*tensor_inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
