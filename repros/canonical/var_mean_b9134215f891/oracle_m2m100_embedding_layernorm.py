"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 scaled token embedding, computed-position embedding gather, population var_mean LayerNorm, affine epilogue, and three aliasing [8192,1024] view returns in one Triton row kernel, whereas Inductor currently materializes the gathered embedding/add region before lowering the 1024-wide norm template and view returns; Inductor cannot do this today because the scheduler/norm canonicalization does not sink row-local embedding and computed-position gathers through the add into the fixed-hidden reduction loop; the fix is SCHEDULER_FUSION: allow rowwise gather/add producers with matching [rows, hidden] iteration space and scalar row index arithmetic to fuse into generated layernorm codegen."""
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


BATCH = 64
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 1024
TOKEN_VOCAB = 128112
POSITION_VOCAB = 1026
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
VIEW_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-5
EMBED_SCALE = 32.0
OUTPUT_COUNT = 3
BLOCK_H = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _m2m100_embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        cumsum_ptr,
        position_mask_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        embed_scale: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        cumsum_i32 = tl.load(cumsum_ptr + row).to(tl.int32)
        position_mask = tl.load(position_mask_ptr + row)
        position_id = (cumsum_i32 * position_mask).to(tl.int64) + 1

        token = tl.load(
            token_table_ptr + token_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = tl.where(mask, token * embed_scale + position, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(out_ptr + row * hidden + cols, y, mask=mask)


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
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = _shape_tuple(value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


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
    tuple[tuple[int, int], ...],
]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    token_table = _require_tensor("arg2_1", inputs[0], (TOKEN_VOCAB, HIDDEN), torch.float32)
    token_ids = _require_tensor("arg1_1", inputs[1], (BATCH, SEQ_LEN), torch.int64)
    cumsum = _require_tensor("cumsum_1", inputs[2], (BATCH, SEQ_LEN), torch.int64)
    position_mask = _require_tensor(
        "convert_element_type_3",
        inputs[3],
        (BATCH, SEQ_LEN),
        torch.int32,
    )
    position_table = _require_tensor(
        "arg198_1",
        inputs[4],
        (POSITION_VOCAB, HIDDEN),
        torch.float32,
    )
    weight = _require_tensor("arg199_1", inputs[5], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg200_1", inputs[6], (HIDDEN,), torch.float32)

    _require_shape("_shape_param_0", inputs[7], BASE_SHAPE)
    output_shapes = tuple(
        _require_shape(f"_shape_param_{index}", inputs[index + 7], VIEW_SHAPE)
        for index in range(1, OUTPUT_COUNT + 1)
    )

    tensor_inputs = (token_table, token_ids, cumsum, position_mask, position_table, weight, bias)
    if any(value.device != token_table.device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return token_table, token_ids, cumsum, position_mask, position_table, weight, bias, output_shapes


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the complete M2M100 embedding LayerNorm alias scope.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns
    the same three float32 [8192,1024] views. The returned tensors are views
    of one contiguous [64,128,1024] base buffer, matching eager alias/layout
    behavior for the repeated final views.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_m2m100_embedding_layernorm.py")

    (
        token_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        weight,
        bias,
        output_shapes,
    ) = _validate_inputs(inputs)

    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=token_table.device,
        dtype=torch.float32,
    )
    _m2m100_embedding_layernorm_kernel[(ROWS,)](
        token_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        weight,
        bias,
        out_base,
        hidden=HIDDEN,
        eps=EPS,
        embed_scale=EMBED_SCALE,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return tuple(out_base.view(shape) for shape in output_shapes)


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
