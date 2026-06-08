"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle keeps the selected-add row tile live through population var_mean, rsqrt normalization, affine scale/bias, and the final view in one Triton row kernel, whereas Inductor currently emits one generic fused Welford reduction kernel but reloads and recomputes `add_91 + select(view_as_real_11, 3, 0)` for the normalization epilogue; Inductor cannot do this today because the generic reduction scheduler does not retain a full fixed-width reduction tile's producer values across a multi-output Welford reduction for downstream row-wise consumers; the fix is SCHEDULER_FUSION: teach the row-reduction schedule/template to preserve fixed hidden-size producer tiles across var_mean and directly feed the affine epilogue when register pressure is acceptable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
    libdevice = None

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


# --- Oracle kernel(s) ---

ROWS = 16384
HIDDEN = 768
EPS = 1.0e-12
BLOCK_M = 2

if triton is not None:

    @triton.jit
    def _select_add_layernorm_kernel(
        view_real_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows_total: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]
        mask = (rows < rows_total) & (cols < hidden)

        flat_offsets = rows * hidden + cols
        add_values = tl.load(add_ptr + flat_offsets, mask=mask, other=0.0)
        selected = tl.load(view_real_ptr + (flat_offsets * 2), mask=mask, other=0.0)
        values = add_values + selected
        values = tl.where(mask, values, 0.0)

        mean_vec = values
        m2_vec = tl.full([BLOCK_M, BLOCK_N], 0.0, tl.float32)
        weight_vec = tl.where(mask, 1.0, 0.0)
        mean, m2, _ = triton_helpers.welford(mean_vec, m2_vec, weight_vec, 1)
        variance = m2 / hidden
        invstd = libdevice.rsqrt(variance + eps)[:, None]

        gamma = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        beta = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        out = ((values - mean[:, None]) * invstd) * gamma + beta
        tl.store(out_ptr + flat_offsets, out, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    view_real, add_input, weight, bias, shape_param = inputs
    tensor_inputs = (view_real, add_input, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected = (
        ((32, 512, HIDDEN, 2), (786432, 1536, 2, 1)),
        ((32, 512, HIDDEN), (393216, HIDDEN, 1)),
        ((HIDDEN,), (1,)),
        ((HIDDEN,), (1,)),
    )
    for index, (tensor, (shape, stride)) in enumerate(zip(tensor_inputs, expected)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(tensor.shape)} != {shape}")
        if tuple(tensor.stride()) != stride:
            raise ValueError(f"input {index} stride {tuple(tensor.stride())} != {stride}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {tensor.dtype} != torch.float32")
        if not tensor.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    if [int(dim) for dim in shape_param] != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected output view shape parameter: {shape_param!r}")

    return view_real, add_input, weight, bias, shape_param


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
        raise RuntimeError("Triton is required for oracle_fused_select_layernorm.py")

    view_real, add_input, weight, bias, shape_param = _validate_inputs(inputs)
    out_shape = tuple(int(dim) for dim in shape_param)
    out = torch.empty_strided(
        out_shape,
        (HIDDEN, 1),
        device=add_input.device,
        dtype=torch.float32,
    )
    _select_add_layernorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        view_real,
        add_input,
        weight,
        bias,
        out,
        rows_total=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
        num_warps=4,
    )
    return out


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
