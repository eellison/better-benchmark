"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 token embedding gather, fp32 hidden-size-512 RMSNorm mean(square)+rsqrt eps=1e-6, fp32 affine weight multiply, and three returned [4096,512] views from one shape-specialized Triton row kernel over a single result storage, whereas Inductor lowers the embedding producer, generic mean reduction, RMSNorm epilogue, and aliasing view returns through its general reduction scheduling path; Inductor cannot do this today because norm-template canonicalization does not recognize a direct token embedding gather feeding fixed-K RMSNorm with multi-view alias outputs as one semantic inference pattern; the fix is NEW_PATTERN: add an embedding-RMSNorm alias template that folds indexed embedding loads, fp32 row RMS reduction, weight epilogue, and sibling view returns into one specialized lowering."""
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
BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 512
VOCAB = 250112
INPUT_ID_SHAPE = (BATCH, SEQ)
BASE_SHAPE = (BATCH, SEQ, HIDDEN)
BASE_STRIDE = (SEQ * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-6
BLOCK_M = 4

if triton is not None:

    @triton.jit
    def _embedding_rmsnorm_kernel(
        embedding_ptr,
        input_ids_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)

        token_ids = tl.load(input_ids_ptr + rows)
        x = tl.load(
            embedding_ptr + token_ids[:, None] * hidden + cols[None, :],
        ).to(tl.float32)

        mean_square = tl.sum(x * x, axis=1) / hidden
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        y = x * inv_rms[:, None] * weight[None, :]

        tl.store(
            out_ptr + rows[:, None] * hidden + cols[None, :],
            y,
        )


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    embedding, input_ids, weight, shape0, shape1, shape2 = inputs
    embedding_t = _require_tensor("arg1_1", embedding, (VOCAB, HIDDEN), torch.float32)
    input_ids_t = _require_tensor("arg0_1", input_ids, INPUT_ID_SHAPE, torch.int64)
    weight_t = _require_tensor("arg2_1", weight, (HIDDEN,), torch.float32)

    if input_ids_t.device != embedding_t.device or weight_t.device != embedding_t.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    out_shape0 = _require_shape("_shape_param_0", shape0, VIEW_SHAPE)
    out_shape1 = _require_shape("_shape_param_1", shape1, VIEW_SHAPE)
    out_shape2 = _require_shape("_shape_param_2", shape2, VIEW_SHAPE)
    return embedding_t, input_ids_t, weight_t, out_shape0, out_shape1, out_shape2


@oracle_impl(hardware="H100", shapes="(T([250112, 512], f32), T([32, 128], i64, gen=Index(250112)), T([512], f32), S([4096, 512]), S([4096, 512]), S([4096, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full-scope MT5 embedding + RMSNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three f32 [4096,512] view outputs. The returned views alias one
    [32,128,512] base buffer, matching the eager `view(mul_tensor_1, shape)`
    contract.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_rmsnorm_aliases.py")

    embedding, input_ids, weight, shape0, shape1, shape2 = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=embedding.device,
        dtype=torch.float32,
    )

    _embedding_rmsnorm_kernel[(ROWS // BLOCK_M,)](
        embedding,
        input_ids,
        weight,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_m=BLOCK_M,
        block_h=HIDDEN,
        num_warps=8,
        num_stages=3,
    )

    return (base.view(shape0), base.view(shape1), base.view(shape2))


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
