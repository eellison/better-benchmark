"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle algebraically eliminates the captured `rand > 1e-30` dropout mask and `* 1.0` scale before computing the complete residual LayerNorm transpose plus `rsqrt/768` side output in one row Triton kernel, whereas Inductor currently carries the input-seeded RNG, comparison, and no-op dropout multiply through the normalization and transpose schedule; Inductor cannot do this today because dropout decomposition is treated as a stochastic producer and the algebraic simplifier does not prove this near-zero-probability unit-scale mask to be identity before codegen; the fix is ALGEBRAIC_ELIMINATION: canonicalize dropout probabilities that round to zero with unit scale, delete the RNG/mask/no-op multiply chain, and emit the residual LayerNorm plus transpose and side-output stores directly."""
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
ROWS = 32 * 512
BATCH = 32
SEQ_LEN = 512
HIDDEN = 768
BLOCK_H = 1024
EPS = 1.0e-5

if triton is not None:

    @triton.jit
    def _dropout_layernorm_transpose_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        transposed_out_ptr,
        inv_div_out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        x = residual + addmm
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        inv_std = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normalized = centered * inv_std * weight + bias

        # transposed_out has shape [HIDDEN, ROWS] and stride [1, HIDDEN], so
        # row-major stores into its backing storage implement the permute view.
        tl.store(transposed_out_ptr + offsets, normalized, mask=mask)
        tl.store(inv_div_out_ptr + row, inv_std / hidden)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _expect_tensor(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm, seeds, residual, weight, bias, shape0, shape1 = inputs
    addmm_t = _expect_tensor("addmm_19", addmm, (ROWS, HIDDEN), torch.float32)
    seeds_t = _expect_tensor("inductor_seeds", seeds, (13,), torch.int64)
    residual_t = _expect_tensor(
        "add_36", residual, (BATCH, SEQ_LEN, HIDDEN), torch.float32
    )
    weight_t = _expect_tensor("arg63_1", weight, (HIDDEN,), torch.float32)
    bias_t = _expect_tensor("arg64_1", bias, (HIDDEN,), torch.float32)

    device = addmm_t.device
    if any(t.device != device for t in (seeds_t, residual_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(shape0) != (BATCH, SEQ_LEN, HIDDEN):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (-1, HIDDEN):
        raise ValueError(f"unexpected final view shape parameter: {shape1!r}")

    return addmm_t, seeds_t, residual_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([13], i64), T([32, 512, 768], f32), T([768], f32), T([768], f32), S([32, 512, 768]), S([-1, 768]))")
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
        raise RuntimeError("Triton is required for oracle_dropout_layernorm_transpose.py")

    addmm, _seeds, residual, weight, bias = _validate_inputs(inputs)
    transposed_out = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=addmm.device,
        dtype=torch.float32,
    )
    inv_div_out = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )

    _dropout_layernorm_transpose_kernel[(ROWS,)](
        addmm,
        residual,
        weight,
        bias,
        transposed_out,
        inv_div_out,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=2,
        num_stages=1,
    )
    return (transposed_out, inv_div_out)


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
        print(
            f"NOTE: {REPRO_ID} contains input-seeded Inductor RNG, but this "
            "oracle eliminates the captured identity dropout mask"
        )

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
