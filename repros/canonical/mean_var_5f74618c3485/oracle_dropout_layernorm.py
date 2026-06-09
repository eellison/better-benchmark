"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT training dropout-add-dropout-layernorm scope in one fixed-width Triton row kernel, including both Inductor-seeded dropout masks, unbiased variance over hidden size 768, affine scale/bias, and the final [16384,768] view, whereas tuned Inductor's norm-template lowering is in the same mandatory memory-traffic and row-reduction envelope for this small hidden dimension; Inductor cannot materially improve it through local fusion once the stochastic producers and normalization are already scheduled around the same required input reads, one row reduction, rsqrt/sqrt, and output store; the fix is BANDWIDTH_BOUND, but exact stochastic value equality with eager is not established, so this remains an at-floor structural diagnostic rather than a true_floor proof."""
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
N_ROWS = 16_384
HIDDEN = 768
BLOCK_H = 1024
SEED_INDEX_0 = 54
SEED_INDEX_1 = 55
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6

if triton is not None:

    @triton.jit
    def _dropout_layernorm_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        seed_index_0: tl.constexpr,
        seed_index_1: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        var_correction: tl.constexpr,
        denom_eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        seed0 = tl.load(seeds_ptr + seed_index_0)
        seed1 = tl.load(seeds_ptr + seed_index_1)
        random0 = tl.rand(seed0, offsets.to(tl.uint32))
        random1 = tl.rand(seed1, offsets.to(tl.uint32))

        dropped0 = tl.where(random0 > dropout_p, addmm * dropout_scale, 0.0)
        added = residual + dropped0
        x = tl.where(random1 > dropout_p, added * dropout_scale, 0.0)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / (
            hidden - var_correction
        )
        denom = tl.sqrt(tl.maximum(variance, 0.0)) + denom_eps

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (weight * centered) / denom + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm, seeds, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm, seeds, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (N_ROWS, HIDDEN),
        (61,),
        (128, 128, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(
                f"input {index} must be contiguous, got stride={value.stride()}"
            )

    if addmm.dtype != torch.float32 or residual.dtype != torch.float32:
        raise TypeError("activation inputs must be torch.float32")
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("layernorm weight and bias must be torch.float32")
    if seeds.dtype != torch.int64:
        raise TypeError(f"inductor seeds must be torch.int64, got {seeds.dtype}")

    if _shape_tuple(shape0) != (128, 128, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (N_ROWS, HIDDEN):
        raise ValueError(f"unexpected final view shape parameter: {shape1!r}")

    return addmm, seeds, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([61], i64), T([128, 128, 768], f32), T([768], f32), T([768], f32), S([128, 128, 768]), S([16384, 768]))")
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
        raise RuntimeError("Triton is required for oracle_dropout_layernorm.py")

    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (N_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )

    _dropout_layernorm_kernel[(N_ROWS,)](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        seed_index_0=SEED_INDEX_0,
        seed_index_1=SEED_INDEX_1,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        var_correction=VAR_CORRECTION,
        denom_eps=DENOM_EPS,
        block_h=BLOCK_H,
        num_warps=2,
        num_stages=1,
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
