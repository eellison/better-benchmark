"""Gap diagnosis (classification: BANDWIDTH_BOUND): this full-scope oracle
computes the complete GPT-J/GPT-Neo GELU-backward pointwise expression once,
stores the live transposed-view output layout, and accumulates the sibling
column-sum output in the same Triton pass, but local CUDAGraph timing is at
floor versus Inductor for this shape; the remaining cost is dominated by
streaming two f32 inputs, storing the full f32 tensor output, tanh/SFU work, and
the unavoidable column reduction, so this does not prove a scheduler/codegen
gap; the fix is BANDWIDTH_BOUND: no Inductor change is justified by this oracle
unless a broader exact algebraic or approximation opportunity is introduced."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
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


ROWS = 128
COLS = 16384
OUT0_SHAPE = (COLS, ROWS)
OUT0_STRIDE = (1, COLS)
OUT1_SHAPE = (COLS,)


def _expect_f32_contiguous(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(value.stride())}")
    return value

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_COLS": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_COLS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_COLS": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_COLS": 32}, num_warps=8, num_stages=3),
        ],
        key=["N_ROWS", "N_COLS"],
    )
    @triton.jit
    def _gelu_backward_sum_kernel(
        grad_ptr,
        x_ptr,
        transpose_out_ptr,
        sum_out_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        rows = tl.arange(0, N_ROWS)
        offsets = rows[:, None] * N_COLS + cols[None, :]
        mask = cols[None, :] < N_COLS

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = x * 0.5
        mul_tensor_1 = grad * mul_tensor
        pow_tensor_scalar = x * x * x
        mul_tensor_2 = pow_tensor_scalar * 0.044715
        add_tensor = x + mul_tensor_2
        mul_tensor_3 = add_tensor * 0.7978845608028654
        tanh_default = libdevice.tanh(mul_tensor_3)
        add_tensor_1 = tanh_default + 1.0
        mul_tensor_4 = grad * add_tensor_1
        mul_tensor_5 = tanh_default * tanh_default
        sub_tensor = 1.0 - mul_tensor_5
        mul_tensor_6 = mul_tensor_1 * sub_tensor
        mul_tensor_7 = mul_tensor_6 * 0.7978845608028654
        mul_tensor_8 = mul_tensor_7 * 0.044715
        pow_tensor_scalar_1 = x * x
        mul_scalar = pow_tensor_scalar_1 * 3.0
        mul_tensor_9 = mul_tensor_8 * mul_scalar
        add_tensor_2 = mul_tensor_7 + mul_tensor_9
        mul_tensor_10 = mul_tensor_4 * 0.5
        out = add_tensor_2 + mul_tensor_10

        tl.store(transpose_out_ptr + offsets, out, mask=mask)
        col_sums = tl.sum(out, axis=0)
        tl.store(sum_out_ptr + cols, col_sums, mask=cols < N_COLS)


@oracle_impl(hardware="H100", shapes="(T([128, 16384], f32), T([128, 16384], f32), S([1, 128, 16384]), S([1, 128, 16384]), S([128, 16384]), S([16384]))")
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
        raise RuntimeError("Triton is required for this oracle")

    mm_326, arg214_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    _expect_f32_contiguous("mm_326", mm_326, (ROWS, COLS))
    _expect_f32_contiguous("arg214_1", arg214_1, (ROWS, COLS))
    if _shape_param_0 != [1, ROWS, COLS] or _shape_param_1 != [1, ROWS, COLS]:
        raise ValueError("unexpected 3D view shape parameters")
    if _shape_param_2 != [ROWS, COLS] or _shape_param_3 != [COLS]:
        raise ValueError("unexpected output view shape parameters")

    transpose_out = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=mm_326.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty(OUT1_SHAPE, device=mm_326.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_COLS"]),)
    _gelu_backward_sum_kernel[grid](
        mm_326,
        arg214_1,
        transpose_out,
        sum_out,
        N_ROWS=ROWS,
        N_COLS=COLS,
    )
    return transpose_out, sum_out


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
