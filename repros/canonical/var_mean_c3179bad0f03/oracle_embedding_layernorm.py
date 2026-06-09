"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NanoGPT token embedding plus generated position embedding and fp32 hidden-size-768 LayerNorm in one shape-specialized Triton row kernel, preserving `var_mean(..., correction=0, keepdim=True)`, eps=1e-5 affine epilogue, and final `[64, 768]` view output, whereas tuned Inductor already lowers this fixed-shape normalization region to the same single-kernel launch and mandatory memory-traffic envelope; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is dominated by required indexed token/position/affine reads, one small row reduction, rsqrt latency, output stores, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding-layernorm case unless broader normalization-template or launch-overhead work moves the whole family."""
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

SEQ_LEN = 64
HIDDEN = 768
WORD_VOCAB = 50304
POSITION_VOCAB = 1024
EPS = 1.0e-5
BLOCK_H = 1024
DEFAULT_NUM_WARPS = 1

if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        word_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        col_mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)

        word = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + row * hidden + cols,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        x = tl.where(col_mask, word + position, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        centered_for_var = tl.where(col_mask, centered, 0.0)
        variance = tl.sum(centered_for_var * centered_for_var, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y, mask=col_mask)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    word_table, token_ids, position_table, weight, bias, out_shape = inputs
    tensor_inputs = (word_table, token_ids, position_table, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (1, SEQ_LEN),
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
        raise TypeError("LayerNorm weight and bias must be torch.float32")
    if token_ids.dtype != torch.int64:
        raise TypeError(f"token ids must be torch.int64, got {token_ids.dtype}")

    out_shape_tuple = tuple(int(dim) for dim in out_shape)
    if out_shape_tuple != (SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected output shape parameter: {out_shape!r}")

    return word_table, token_ids, position_table, weight, bias, out_shape_tuple


def oracle_embedding_layernorm(
    word_table: torch.Tensor,
    token_ids: torch.Tensor,
    position_table: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out_shape: tuple[int, int],
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """Compute the complete Repro.forward output with one Triton row kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm.py")

    base = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=word_table.dtype,
    )
    _embedding_layernorm_kernel[(SEQ_LEN,)](
        word_table,
        token_ids,
        position_table,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return base.view(out_shape)


@oracle_impl(hardware="H100", shapes="(T([50304, 768], f32), T([1, 64], i64, gen=Index(50304)), T([1024, 768], f32), T([768], f32), T([768], f32), S([64, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward embedding + LayerNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single f32 `[64, 768]` output with contiguous `(768, 1)` stride.
    """
    return oracle_embedding_layernorm(*_validate_inputs(inputs))


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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
