"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT embedding-position-token-type-add, Inductor-seeded dropout, unbiased hidden-size-768 LayerNorm, affine scale/bias, and final [16384,768] view in one shape-specialized Triton row kernel, whereas tuned Inductor reaches the same full-scope row-normalization envelope for this norm-template-canonicalization case; Inductor cannot materially improve it through a local canonicalization today because the remaining work is dominated by the required embedding/position/token-type/weight/bias reads, one stochastic mask, two row reductions, sqrt/divide epilogue, and output store rather than an avoidable scheduler split; the fix is BANDWIDTH_BOUND: record this as at_floor unless a broader memory-traffic, RNG, or launch-overhead improvement appears. Exact stochastic equality is skipped by the harness, so the floor status is not_true_floor."""
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
VOCAB = 20_005
BATCH = 128
SEQ_LEN = 128
POSITION_SEQ_LEN = 512
TOKEN_TYPES = 3
HIDDEN = 768
ROWS = BATCH * SEQ_LEN
WORD_EMBEDDING_SHAPE = (VOCAB, HIDDEN)
INPUT_ID_SHAPE = (BATCH, SEQ_LEN)
POSITION_SHAPE = (1, POSITION_SEQ_LEN, HIDDEN)
TOKEN_TYPE_SHAPE = (TOKEN_TYPES, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 1024
SEED_COUNT = 61
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6
FLOOR_STATUS = "not_true_floor"

if triton is not None:

    @triton.jit
    def _bert_embedding_dropout_layernorm_kernel(
        word_embedding_ptr,
        input_ids_ptr,
        position_embedding_ptr,
        token_type_embedding_ptr,
        token_type_ids_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        seed_index: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        var_correction: tl.constexpr,
        denom_eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden
        offsets = row * hidden + cols
        seq_index = row % seq_len

        word_id = tl.load(input_ids_ptr + row)
        token_type_id = tl.load(token_type_ids_ptr + row)

        word = tl.load(
            word_embedding_ptr + word_id * hidden + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + seq_index * hidden + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        token_type = tl.load(
            token_type_embedding_ptr + token_type_id * hidden + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, offsets.to(tl.uint32))
        x = tl.where(random > dropout_p, (word + position + token_type) * dropout_scale, 0.0)
        x_for_reduce = tl.where(col_mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(col_mask, centered * centered, 0.0), axis=0) / (
            hidden - var_correction
        )
        denom = tl.sqrt(tl.maximum(variance, 0.0)) + denom_eps

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = (weight * centered) / denom + bias
        tl.store(out_ptr + offsets, out, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


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
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        word_embedding,
        input_ids,
        position_embedding,
        token_type_embedding,
        token_type_ids,
        weight,
        bias,
        output_shape,
    ) = inputs

    word_embedding_t = _require_tensor(
        "arg1_1", word_embedding, WORD_EMBEDDING_SHAPE, torch.float32
    )
    input_ids_t = _require_tensor("arg0_1", input_ids, INPUT_ID_SHAPE, torch.int64)
    position_embedding_t = _require_tensor(
        "arg2_1", position_embedding, POSITION_SHAPE, torch.float32
    )
    token_type_embedding_t = _require_tensor(
        "arg3_1", token_type_embedding, TOKEN_TYPE_SHAPE, torch.float32
    )
    token_type_ids_t = _require_tensor(
        "arg4_1", token_type_ids, INPUT_ID_SHAPE, torch.int64
    )
    weight_t = _require_tensor("arg5_1", weight, (HIDDEN,), torch.float32)
    bias_t = _require_tensor("arg6_1", bias, (HIDDEN,), torch.float32)

    device = word_embedding_t.device
    tensor_inputs = (
        input_ids_t,
        position_embedding_t,
        token_type_embedding_t,
        token_type_ids_t,
        weight_t,
        bias_t,
    )
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    shape = _shape_tuple(output_shape)
    if shape != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape!r}")

    return (
        word_embedding_t,
        input_ids_t,
        position_embedding_t,
        token_type_embedding_t,
        token_type_ids_t,
        weight_t,
        bias_t,
        shape,
    )


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
        raise RuntimeError("Triton is required for oracle_bert_embedding_dropout_layernorm.py")

    (
        word_embedding,
        input_ids,
        position_embedding,
        token_type_embedding,
        token_type_ids,
        weight,
        bias,
        output_shape,
    ) = _validate_inputs(inputs)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, word_embedding.device)
    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )

    _bert_embedding_dropout_layernorm_kernel[(ROWS,)](
        word_embedding,
        input_ids,
        position_embedding,
        token_type_embedding,
        token_type_ids,
        weight,
        bias,
        seeds,
        output,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        seed_index=SEED_INDEX,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        var_correction=VAR_CORRECTION,
        denom_eps=DENOM_EPS,
        block_h=BLOCK_H,
        num_warps=2,
        num_stages=1,
    )
    return output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if FLOOR_STATUS == "not_true_floor":
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
