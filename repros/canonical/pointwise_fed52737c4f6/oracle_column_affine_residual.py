"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT residual column-affine pointwise scope in one shape-specialized Triton pass, including the metadata-only `[32768, 128] <-> [256, 128, 128]` views, the two fp32 per-column affine stages, and the fresh contiguous `[32768, 128]` output, whereas Inductor already lowers the same captured graph to a single fused pointwise kernel over the required output; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is dominated by mandatory reads of the two activation matrices, tiny reused column vectors, and one output store; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise bandwidth case unless broader pointwise memory/codegen improvements move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


ROWS = 32768
COLS = 128
MATRIX_SHAPE = (ROWS, COLS)
MATRIX_STRIDE = (COLS, 1)
VECTOR_SHAPE = (COLS,)
VIEW_SHAPE = (256, 128, 128)
OUTPUT_SHAPE = MATRIX_SHAPE
OUTPUT_STRIDE = MATRIX_STRIDE


# --- Oracle kernel(s) ---

if triton is not None:

    # Preserve the repro's separate fp32 aten mul/add rounding boundaries.
    @triton.jit
    def _f32_mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=4),
        ],
        key=[],
    )
    @triton.jit
    def _column_affine_residual_kernel(
        addmm0_ptr,
        addmm1_ptr,
        scale0_ptr,
        bias0_ptr,
        scale1_ptr,
        bias1_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, N_COLS)
        offsets = rows[:, None] * N_COLS + cols[None, :]
        mask = rows[:, None] < N_ROWS

        addmm0 = tl.load(addmm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        addmm1 = tl.load(addmm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale0 = tl.load(scale0_ptr + cols).to(tl.float32)[None, :]
        bias0 = tl.load(bias0_ptr + cols).to(tl.float32)[None, :]
        scale1 = tl.load(scale1_ptr + cols).to(tl.float32)[None, :]
        bias1 = tl.load(bias1_ptr + cols).to(tl.float32)[None, :]

        first_affine = _f32_mul(addmm1, scale0)
        first_affine = _f32_add(first_affine, bias0)
        residual = _f32_add(addmm0, first_affine)
        second_affine = _f32_mul(residual, scale1)
        out = _f32_add(second_affine, bias1)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(name: str, value, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype must be torch.float32, got {value.dtype}")
    if value.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_column_affine_residual.py")
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    addmm0, addmm1, scale0, bias0, scale1, bias1, shape0, shape1, shape2 = inputs
    addmm0 = _require_tensor("addmm_350", addmm0, MATRIX_SHAPE)
    addmm1 = _require_tensor("addmm_351", addmm1, MATRIX_SHAPE)
    scale0 = _require_tensor("arg1069_1", scale0, VECTOR_SHAPE)
    bias0 = _require_tensor("arg1070_1", bias0, VECTOR_SHAPE)
    scale1 = _require_tensor("arg1083_1", scale1, VECTOR_SHAPE)
    bias1 = _require_tensor("arg1084_1", bias1, VECTOR_SHAPE)

    if tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape: {shape0}")
    if tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"unexpected second view shape: {shape1}")
    if tuple(shape2) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape: {shape2}")

    return addmm0, addmm1, scale0, bias0, scale1, bias1


@oracle_impl(hardware="H100", shapes="(T([32768, 128], f32), T([32768, 128], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), S([256, 128, 128]), S([256, 128, 128]), S([32768, 128]))")
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
    addmm0, addmm1, scale0, bias0, scale1, bias1 = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm0.device,
        dtype=addmm0.dtype,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _column_affine_residual_kernel[grid](
        addmm0,
        addmm1,
        scale0,
        bias0,
        scale1,
        bias1,
        output,
        N_ROWS=ROWS,
        N_COLS=COLS,
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
