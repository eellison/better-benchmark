"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured GELU-backward producer and contiguous f32[3072] column-sum output with a direct two-stage Triton row-split reduction, whereas Inductor already fuses the same pointwise producer into a two-stage reduction over the required f32 input streams; Inductor cannot materially avoid the dominant input traffic, libdevice transcendental work, f32 accumulation, or small final reduction for this isolated scope; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless broader exact pointwise-transcendental codegen or reduction throughput improvements move both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


ROWS = 16384
COLS = 3072
VIEW_SHAPE = (32, 512, COLS)
FLAT_SHAPE = (ROWS, COLS)
OUT_SHAPE = (COLS,)
ROWS_PER_GROUP = 373
NUM_GROUPS = 44
PARTIAL_XNUMEL = NUM_GROUPS * COLS
FINAL_X_BLOCK = 32
FINAL_R_BLOCK = 64


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
            triton.Config({"X_BLOCK": 32, "R_BLOCK": 8}, num_warps=4, num_stages=1),
            triton.Config({"X_BLOCK": 64, "R_BLOCK": 8}, num_warps=4, num_stages=1),
            triton.Config({"X_BLOCK": 128, "R_BLOCK": 8}, num_warps=4, num_stages=1),
            triton.Config({"X_BLOCK": 128, "R_BLOCK": 8}, num_warps=8, num_stages=1),
            triton.Config({"X_BLOCK": 256, "R_BLOCK": 8}, num_warps=8, num_stages=1),
        ],
        key=["ROWS_", "COLS_"],
    )
    @triton.jit
    def _gelu_backward_partial_sum_kernel(
        grad_ptr,
        x_ptr,
        partials_ptr,
        XNUMEL_: tl.constexpr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        ROWS_PER_GROUP_: tl.constexpr,
        X_BLOCK: tl.constexpr,
        R_BLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
        group = xindex // COLS_
        col = xindex % COLS_
        rbase = tl.arange(0, R_BLOCK)[None, :]
        acc = tl.zeros((X_BLOCK, R_BLOCK), dtype=tl.float32)

        for roffset in tl.range(0, ROWS_PER_GROUP_, R_BLOCK):
            rindex = roffset + rbase
            rows = group * ROWS_PER_GROUP_ + rindex
            row_mask = (rindex < ROWS_PER_GROUP_) & (rows < ROWS_)
            mask = row_mask
            offsets = rows * COLS_ + col

            grad = tl.load(
                grad_ptr + offsets,
                mask=mask,
                eviction_policy="evict_first",
                other=0.0,
            ).to(tl.float32)
            x = tl.load(
                x_ptr + offsets,
                mask=mask,
                eviction_policy="evict_first",
                other=0.0,
            ).to(tl.float32)

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
            acc = tl.where(row_mask, acc + out, acc)

        partial = tl.sum(acc, axis=1)[:, None]
        tl.store(partials_ptr + xindex, partial)

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_ptr,
        NUM_GROUPS_: tl.constexpr,
        COLS_: tl.constexpr,
        X_BLOCK_: tl.constexpr,
        R_BLOCK_: tl.constexpr,
    ):
        cols = tl.program_id(0) * X_BLOCK_ + tl.arange(0, X_BLOCK_)[:, None]
        groups = tl.arange(0, R_BLOCK_)[None, :]
        mask = groups < NUM_GROUPS_
        vals = tl.load(
            partials_ptr + groups * COLS_ + cols,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(vals, axis=1)[:, None]
        tl.store(out_ptr + cols, sums)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _expect_f32_contiguous(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != FLAT_SHAPE:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {FLAT_SHAPE}")
    if tuple(value.stride()) != (COLS, 1):
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {(COLS, 1)}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    grad, x, shape0, shape1, shape2, shape3 = inputs
    grad = _expect_f32_contiguous("mm_42", grad)
    x = _expect_f32_contiguous("arg55_1", x)

    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape0!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_1 is {shape1!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(shape2) != FLAT_SHAPE:
        raise ValueError(f"_shape_param_2 is {shape2!r}, expected {FLAT_SHAPE}")
    if _shape_tuple(shape3) != OUT_SHAPE:
        raise ValueError(f"_shape_param_3 is {shape3!r}, expected {OUT_SHAPE}")
    return grad, x


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
        raise RuntimeError("Triton is required for oracle_gelu_backward_sum.py")

    grad, x = _validate_inputs(inputs)
    partials = torch.empty(
        (NUM_GROUPS, COLS),
        device=grad.device,
        dtype=torch.float32,
    )
    output = torch.empty(OUT_SHAPE, device=grad.device, dtype=torch.float32)

    partial_grid = lambda meta: (triton.cdiv(PARTIAL_XNUMEL, meta["X_BLOCK"]),)
    _gelu_backward_partial_sum_kernel[partial_grid](
        grad,
        x,
        partials,
        XNUMEL_=PARTIAL_XNUMEL,
        ROWS_=ROWS,
        COLS_=COLS,
        ROWS_PER_GROUP_=ROWS_PER_GROUP,
    )
    _finalize_partials_kernel[(triton.cdiv(COLS, FINAL_X_BLOCK),)](
        partials,
        output,
        NUM_GROUPS_=NUM_GROUPS,
        COLS_=COLS,
        X_BLOCK_=FINAL_X_BLOCK,
        R_BLOCK_=FINAL_R_BLOCK,
        num_warps=8,
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
