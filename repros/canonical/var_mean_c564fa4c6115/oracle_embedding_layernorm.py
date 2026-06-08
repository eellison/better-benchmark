"""Gap diagnosis (classification: NEW_PATTERN): this oracle emits a minimal full-scope Longformer embedding-plus-LayerNorm row kernel with direct word, masked-cumsum position, and token-type table loads followed by population var_mean(correction=0, keepdim=True), eps=1e-5 rsqrt, affine scale, and the final contiguous f32 store, whereas Inductor currently emits a single generic persistent-reduction kernel for the full graph with indirect-index assertions, generic reduction bookkeeping, and nonsemantic normalization codegen; Inductor cannot do this today because its pattern library has no semantic embedding-layernorm lowering that specializes the integer position-id arithmetic and elides generic indexed-reduction scaffolding; the fix is NEW_PATTERN: add a Longformer-style embedding-layernorm pattern that lowers the indexed table loads, exact position arithmetic, and affine normalization epilogue as one specialized row kernel."""
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
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 8
SEQ_LEN = 1024
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
WORD_VOCAB = 50265
POSITION_ROWS = 4098
TOKEN_TYPE_ROWS = 1
BLOCK_H = 1024
EPS = 1.0e-5
OUTPUT_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
OUTPUT_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def oracle_kernel(
        word_table_ptr,
        token_ids_ptr,
        cumsum_ptr,
        position_mask_ptr,
        position_table_ptr,
        token_type_table_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        hidden: tl.constexpr,
        block_h: tl.constexpr,
        eps: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        token_id = tl.load(token_ids_ptr + rows, mask=row_mask, other=0)
        cumsum_i32 = tl.load(cumsum_ptr + rows, mask=row_mask, other=0).to(tl.int32)
        position_mask_i32 = tl.load(position_mask_ptr + rows, mask=row_mask, other=0)
        position_id = (cumsum_i32 * position_mask_i32).to(tl.int64) + 1

        word = tl.load(
            word_table_ptr + token_id * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        token_type = tl.load(
            token_type_table_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        x = (word + position) + token_type
        x_masked = tl.where(mask, x, 0.0)
        mean_vec = tl.sum(x_masked, axis=1) / hidden
        mean = mean_vec[:, None]
        centered = x - mean
        var_vec = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        invstd = tl.rsqrt(var_vec + eps)[:, None]

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(output_ptr + rows * hidden + cols, y, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    (
        word_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        token_type_table,
        weight,
        bias,
    ) = inputs
    tensor_inputs = (
        word_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        token_type_table,
        weight,
        bias,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("all repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        (WORD_VOCAB, HIDDEN),
        (BATCH, SEQ_LEN),
        (BATCH, SEQ_LEN),
        (BATCH, SEQ_LEN),
        (POSITION_ROWS, HIDDEN),
        (TOKEN_TYPE_ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")

    fp32_tensors = (word_table, position_table, token_type_table, weight, bias)
    if not all(value.dtype == torch.float32 for value in fp32_tensors):
        dtypes = [value.dtype for value in fp32_tensors]
        raise TypeError(f"expected float32 data tensors, got {dtypes}")
    if token_ids.dtype != torch.int64 or cumsum.dtype != torch.int64:
        raise TypeError("token ids and cumsum must be int64")
    if position_mask.dtype != torch.int32:
        raise TypeError("position mask must be int32")

    devices = {value.device for value in tensor_inputs}
    if len(devices) != 1:
        raise ValueError(f"all tensor inputs must be on one device, got {devices}")

    return tensor_inputs


def oracle_forward(inputs):
    """Run the complete Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        word_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        token_type_table,
        weight,
        bias,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=word_table.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    oracle_kernel[grid](
        word_table,
        token_ids,
        cumsum,
        position_mask,
        position_table,
        token_type_table,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        block_h=BLOCK_H,
        eps=EPS,
        total_rows=ROWS,
    )
    return output


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
