"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MBart token-embedding plus position-embedding plus two dependent hidden-size-1024 population-variance LayerNorms in one Triton row kernel, returning the three aliasing `[8192,1024]` views from one final buffer instead of exposing the intermediate embedding and first normalized activation, whereas Inductor's tuned lowering already matches this full-scope oracle within the required CUDAGraph harness; Inductor cannot materially improve this repro through a local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or recompute-fusion change because the remaining cost is dominated by mandatory token/position/affine reads, two row reductions, rsqrt/affine epilogues, and the final output write; the fix is BANDWIDTH_BOUND: record this norm-template canonicalization case as at_floor unless broader normalization codegen, launch, or memory-traffic work moves both paths."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


VOCAB = 50_265
BATCH = 8
SEQ_LEN = 1024
POSITION_ROWS = 1026
POSITION_OFFSET = 2
HIDDEN = 1024
ROWS = BATCH * SEQ_LEN
EPS = 1.0e-5
WORD_EMBEDDING_SHAPE = (VOCAB, HIDDEN)
INPUT_ID_SHAPE = (BATCH, SEQ_LEN)
POSITION_EMBEDDING_SHAPE = (POSITION_ROWS, HIDDEN)
VECTOR_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _mbart_dual_layernorm_kernel(
        word_embedding_ptr,
        input_ids_ptr,
        position_embedding_ptr,
        weight0_ptr,
        bias0_ptr,
        weight1_ptr,
        bias1_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        position_offset: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        seq_index = row % seq_len
        token_id = tl.load(input_ids_ptr + row)

        word = tl.load(
            word_embedding_ptr + token_id * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        position = tl.load(
            position_embedding_ptr + (seq_index + position_offset) * hidden + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = word + position

        mean0 = tl.sum(x, axis=0) / hidden
        square_mean0 = tl.sum(x * x, axis=0) / hidden
        centered0 = x - mean0
        var0 = square_mean0 - mean0 * mean0
        invstd0 = tl.rsqrt(var0 + eps)

        weight0 = tl.load(
            weight0_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias0 = tl.load(
            bias0_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered0 * invstd0 * weight0 + bias0

        mean1 = tl.sum(y, axis=0) / hidden
        square_mean1 = tl.sum(y * y, axis=0) / hidden
        centered1 = y - mean1
        var1 = square_mean1 - mean1 * mean1
        invstd1 = tl.rsqrt(var1 + eps)

        weight1 = tl.load(
            weight1_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias1 = tl.load(
            bias1_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)
        out = centered1 * invstd1 * weight1 + bias1
        tl.store(out_ptr + row * hidden + cols, out)


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
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    word_embedding = _require_tensor(
        "arg1_1", inputs[0], WORD_EMBEDDING_SHAPE, torch.float32
    )
    input_ids = _require_tensor("arg0_1", inputs[1], INPUT_ID_SHAPE, torch.int64)
    position_embedding = _require_tensor(
        "arg2_1", inputs[2], POSITION_EMBEDDING_SHAPE, torch.float32
    )
    weight0 = _require_tensor("arg3_1", inputs[3], VECTOR_SHAPE, torch.float32)
    bias0 = _require_tensor("arg4_1", inputs[4], VECTOR_SHAPE, torch.float32)
    weight1 = _require_tensor("arg5_1", inputs[5], VECTOR_SHAPE, torch.float32)
    bias1 = _require_tensor("arg6_1", inputs[6], VECTOR_SHAPE, torch.float32)

    device = word_embedding.device
    tensor_inputs = (input_ids, position_embedding, weight0, bias0, weight1, bias1)
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shapes = tuple(_shape_tuple(value) for value in inputs[7:10])
    if output_shapes != (OUTPUT_SHAPE, OUTPUT_SHAPE, OUTPUT_SHAPE):
        raise ValueError(f"unexpected output view shapes: {output_shapes!r}")

    return (
        word_embedding,
        input_ids,
        position_embedding,
        weight0,
        bias0,
        weight1,
        bias1,
        output_shapes,
    )


def oracle_forward(inputs):
    """Run the complete MBart embedding plus dual LayerNorm scope.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns
    the same three f32 [8192,1024] view outputs. All returned tensors alias one
    contiguous final buffer, matching the eager repeated view contract.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mbart_dual_layernorm.py")

    (
        word_embedding,
        input_ids,
        position_embedding,
        weight0,
        bias0,
        weight1,
        bias1,
        output_shapes,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=word_embedding.device,
        dtype=torch.float32,
    )
    _mbart_dual_layernorm_kernel[(ROWS,)](
        word_embedding,
        input_ids,
        position_embedding,
        weight0,
        bias0,
        weight1,
        bias1,
        output,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        position_offset=POSITION_OFFSET,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return tuple(output.view(shape) for shape in output_shapes)


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
