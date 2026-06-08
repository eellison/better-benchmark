"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full LayoutLM word embedding, learned position embedding, four constant-zero box-coordinate embedding gathers, two constant-zero box-size embedding gathers, token-type embedding, hidden-size-768 population var_mean, eps=1e-12 affine LayerNorm, and three returned [16384,768] alias views in one Triton row kernel, whereas Inductor currently lowers the constant full/select/sub embedding producers and generic var_mean LayerNorm as separate scheduled work with materialized intermediate embedding sums; Inductor cannot do this today because norm-template canonicalization does not recognize the LayoutLM multi-table spatial embedding assembly with constant-zero bounding-box ids as a single row-local LayerNorm producer preserving sibling alias views; the fix is NEW_PATTERN: add a LayoutLM embedding-LayerNorm template that folds the direct indexed loads, constant-zero spatial table rows, row statistics, affine epilogue, and alias-view returns into one specialized lowering."""
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


BATCH = 32
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
BLOCK_H = 1024
WORD_VOCAB = 30_522
POSITION_ROWS = 512
SPATIAL_ROWS = 1024
TOKEN_TYPE_ROWS = 2
EPS = 1.0e-12
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _layoutlm_embedding_layernorm_kernel(
        word_table_ptr,
        word_ids_ptr,
        position_table_ptr,
        position_ids_ptr,
        x_position_table_ptr,
        y_position_table_ptr,
        h_position_table_ptr,
        w_position_table_ptr,
        token_type_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        seq = row % seq_len
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        word_id = tl.load(word_ids_ptr + row)
        position_id = tl.load(position_ids_ptr + seq)

        word = tl.load(
            word_table_ptr + word_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_table_ptr + position_id * hidden + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x_position = tl.load(
            x_position_table_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y_position = tl.load(
            y_position_table_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        h_position = tl.load(
            h_position_table_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w_position = tl.load(
            w_position_table_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        token_type = tl.load(
            token_type_table_ptr + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        x = word + position
        x = x + x_position
        x = x + y_position
        x = x + x_position
        x = x + y_position
        x = x + h_position
        x = x + w_position
        x = tl.where(mask, x + token_type, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
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
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
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
    tuple[int, int],
]:
    if len(inputs) != 14:
        raise ValueError(f"{REPRO_ID} expects 14 inputs, got {len(inputs)}")

    (
        word_table,
        word_ids,
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = inputs

    word_table_t = _require_tensor("arg1_1", word_table, (WORD_VOCAB, HIDDEN), torch.float32)
    word_ids_t = _require_tensor("arg0_1", word_ids, (BATCH, SEQ_LEN), torch.int64)
    position_table_t = _require_tensor("arg3_1", position_table, (POSITION_ROWS, HIDDEN), torch.float32)
    position_ids_t = _require_tensor("arg2_1", position_ids, (1, SEQ_LEN), torch.int64)
    x_position_table_t = _require_tensor("arg4_1", x_position_table, (SPATIAL_ROWS, HIDDEN), torch.float32)
    y_position_table_t = _require_tensor("arg5_1", y_position_table, (SPATIAL_ROWS, HIDDEN), torch.float32)
    h_position_table_t = _require_tensor("arg6_1", h_position_table, (SPATIAL_ROWS, HIDDEN), torch.float32)
    w_position_table_t = _require_tensor("arg7_1", w_position_table, (SPATIAL_ROWS, HIDDEN), torch.float32)
    token_type_table_t = _require_tensor("arg8_1", token_type_table, (TOKEN_TYPE_ROWS, HIDDEN), torch.float32)
    weight_t = _require_tensor("arg9_1", weight, (HIDDEN,), torch.float32)
    bias_t = _require_tensor("arg10_1", bias, (HIDDEN,), torch.float32)

    device = word_table_t.device
    tensor_inputs = (
        word_ids_t,
        position_table_t,
        position_ids_t,
        x_position_table_t,
        y_position_table_t,
        h_position_table_t,
        w_position_table_t,
        token_type_table_t,
        weight_t,
        bias_t,
    )
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2))
    for index, shape in enumerate(output_shapes, start=11):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return (
        word_table_t,
        word_ids_t,
        position_table_t,
        position_ids_t,
        x_position_table_t,
        y_position_table_t,
        h_position_table_t,
        w_position_table_t,
        token_type_table_t,
        weight_t,
        bias_t,
        output_shapes[0],
        output_shapes[1],
        output_shapes[2],
    )


def oracle_forward(inputs):
    """Run the full Repro.forward scope and return the three final aliasing views."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layoutlm_embedding_layernorm.py")

    (
        word_table,
        word_ids,
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=word_table.device,
        dtype=torch.float32,
    )
    _layoutlm_embedding_layernorm_kernel[(ROWS,)](
        word_table,
        word_ids,
        position_table,
        position_ids,
        x_position_table,
        y_position_table,
        h_position_table,
        w_position_table,
        token_type_table,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return (
        torch.ops.aten.view.default(out, shape0),
        torch.ops.aten.view.default(out, shape1),
        torch.ops.aten.view.default(out, shape2),
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
