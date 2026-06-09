"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot token embedding, generated positional embedding add, fp32 hidden-size-2560 var_mean normalization, affine scale/bias, and three returned `[4096,2560]` alias views in one Triton row kernel, whereas Inductor currently lowers the indexed embedding/add producer and generic var_mean LayerNorm template as separate scheduled regions with an intermediate `[32,128,2560]` tensor; Inductor cannot do this today because norm-template scheduling does not sink row-local embedding gathers and generated position indexing into the fixed-hidden reduction loop while preserving multi-view alias metadata; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to fuse gather/add producers into the row reduction epilogue and return alias-only views from the single normalized output buffer."""
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


BATCH = 32
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 2560
TOKEN_VOCAB = 8008
POSITION_ROWS = 128
EPS = 1.0e-5
BLOCK_H = 4096
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        token_table_ptr,
        token_ids_ptr,
        position_table_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seq_len: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        token_id = tl.load(token_ids_ptr + row)
        position_id = row % seq_len
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

        x = tl.where(mask, token * 1.0 + position, 0.0)
        mean = tl.sum(x, axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


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
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out_shape0,
        out_shape1,
        out_shape2,
    ) = inputs

    token_table_t = _require_tensor("arg1_1", token_table, (TOKEN_VOCAB, HIDDEN), torch.float32)
    token_ids_t = _require_tensor("arg0_1", token_ids, (BATCH, SEQ_LEN), torch.int64)
    position_table_t = _require_tensor("arg2_1", position_table, (POSITION_ROWS, HIDDEN), torch.float32)
    weight_t = _require_tensor("arg3_1", weight, (HIDDEN,), torch.float32)
    bias_t = _require_tensor("arg4_1", bias, (HIDDEN,), torch.float32)

    device = token_table_t.device
    if any(value.device != device for value in (token_ids_t, position_table_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    output_shapes = tuple(_shape_tuple(shape) for shape in (out_shape0, out_shape1, out_shape2))
    for index, shape in enumerate(output_shapes, start=5):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return (
        token_table_t,
        token_ids_t,
        position_table_t,
        weight_t,
        bias_t,
        output_shapes[0],
        output_shapes[1],
        output_shapes[2],
    )


@oracle_impl(hardware="H100", shapes="(T([8008, 2560], f32), T([32, 128], i64, gen=Index(8008)), T([128, 2560], f32), T([2560], f32), T([2560], f32), S([4096, 2560]), S([4096, 2560]), S([4096, 2560]))")
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
        raise RuntimeError("Triton is required for oracle_blenderbot_embedding_layernorm_aliases.py")

    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out_shape0,
        out_shape1,
        out_shape2,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=token_table.device,
        dtype=torch.float32,
    )
    _embedding_layernorm_kernel[(ROWS,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        seq_len=SEQ_LEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return out.view(out_shape0), out.view(out_shape1), out.view(out_shape2)


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
