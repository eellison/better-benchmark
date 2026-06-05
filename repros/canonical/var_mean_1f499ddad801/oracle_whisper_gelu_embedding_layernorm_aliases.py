"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Whisper fp16 convolution GELU producer, `[8,384,1500] -> [8,1500,384]` layout canonicalization, generated position embedding add with fp16 rounding, fp32 hidden-size-384 var_mean layernorm, affine epilogue, fp16 cast, and three returned `[12000,384]` views aliasing one result allocation in one shape-specialized Triton row kernel, whereas Inductor already reaches the same launch count and memory-traffic envelope for this fixed Whisper norm-template scope; Inductor cannot materially improve this today because the remaining work is dominated by required erf-GELU math, transposed convolution reads, repeated position reads, one hidden-size-384 row reduction, affine reads, rsqrt, output stores, and launch overhead rather than an actionable missed scheduler fusion or algebraic rewrite; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope Whisper GELU-position-layernorm case unless broader norm-template math-codegen or launch-overhead work moves the family."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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


BATCH = 8
CHANNELS = 384
SEQ_LEN = 1500
ROWS = BATCH * SEQ_LEN
HIDDEN = CHANNELS
BLOCK_H = 512
EPS = 1.0e-5
CLASSIFICATION = "BANDWIDTH_BOUND"


if triton is not None:

    @triton.jit
    def _whisper_gelu_position_layernorm_kernel(
        convolution_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        seq_len: tl.constexpr,
        hidden: tl.constexpr,
        block_h: tl.constexpr,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // seq_len
        token = row - batch * seq_len
        cols = tl.arange(0, block_h)
        mask = cols < hidden

        conv_offsets = (batch * hidden + cols) * seq_len + token
        x = tl.load(convolution_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
        gelu = (x * 0.5) * (tl.erf(x * 0.7071067811865476) + 1.0)
        gelu_f16 = gelu.to(tl.float16)

        position = tl.load(position_ptr + token * hidden + cols, mask=mask, other=0.0)
        norm_input = (gelu_f16 + position).to(tl.float16).to(tl.float32)
        norm_input = tl.where(mask, norm_input, 0.0)

        mean = tl.sum(norm_input, axis=0) / hidden
        centered = norm_input - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * hidden + cols, y.to(tl.float16), mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")

    convolution, position, weight, bias, shape0, shape1, shape2 = inputs
    tensor_inputs = (convolution, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (
        (BATCH, CHANNELS, SEQ_LEN),
        (SEQ_LEN, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if convolution.dtype != torch.float16 or position.dtype != torch.float16:
        raise TypeError("convolution and position inputs must be torch.float16")
    if weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("layernorm weight and bias must be torch.float16")

    view_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2))
    for index, shape in enumerate(view_shapes, start=4):
        if shape != (ROWS, HIDDEN):
            raise ValueError(f"input {index} unexpected shape parameter: {shape!r}")

    return convolution, position, weight, bias, *view_shapes


def oracle_whisper_gelu_position_layernorm(
    convolution: torch.Tensor,
    position: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    shape2: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the complete Whisper GELU + position add + layernorm graph."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    base = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=convolution.device,
        dtype=torch.float16,
    )
    _whisper_gelu_position_layernorm_kernel[(ROWS,)](
        convolution,
        position,
        weight,
        bias,
        base,
        seq_len=SEQ_LEN,
        hidden=HIDDEN,
        block_h=BLOCK_H,
        eps=EPS,
        num_warps=4,
        num_stages=4,
    )

    return base.view(shape0), base.view(shape1), base.view(shape2)


def oracle_forward(inputs):
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    return oracle_whisper_gelu_position_layernorm(*_validate_inputs(inputs))


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
