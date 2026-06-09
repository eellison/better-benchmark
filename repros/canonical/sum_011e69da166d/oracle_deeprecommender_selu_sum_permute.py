"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender SELU-style pointwise producer once, writes the returned f32[197951,1024] permute-view backing storage, and accumulates the sibling f32[197951] column sum in the same Triton kernel using Inductor-matching f32 libdevice exp, whereas Inductor currently schedules the required permuted side output and the column reduction as separate generic work over the same expression; Inductor cannot do this today because its scheduler does not fuse a layout-changing materialized side output with a compatible sibling reduction over that shared producer; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a multi-output pointwise-plus-column-reduction kernel when a required permuted materialization and a reduction consume the same pointwise expression."""
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


ROWS = 1024
COLS = 197951
OUT_BASE_SHAPE = (ROWS, COLS)
OUT_BASE_STRIDE = (COLS, 1)
OUT0_SHAPE = (COLS, ROWS)
OUT0_STRIDE = (1, COLS)
OUT1_SHAPE = (COLS,)
OUT1_STRIDE = (1,)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _deeprecommender_selu_sum_permute_kernel(
        arg18_ptr,
        arg17_ptr,
        out_base_ptr,
        out_sum_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_start = tl.program_id(0) * BLOCK_N
        cols = col_start + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for row_start in tl.range(0, M, BLOCK_M):
            rows = row_start + tl.arange(0, BLOCK_M)
            offsets = rows[:, None] * N + cols[None, :]
            mask = col_mask[None, :]

            arg18 = tl.load(arg18_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            arg17 = tl.load(arg17_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            mul_tensor = arg18 * 1.0
            mul_tensor_1 = mul_tensor * 1.7580993408473766
            mul_tensor_2 = arg17 * 1.0
            exp_default = libdevice.exp(mul_tensor_2)
            mul_tensor_3 = mul_tensor_1 * exp_default
            mul_tensor_4 = arg18 * 1.0507009873554805
            le_scalar = arg17 <= 0.0
            value = tl.where(le_scalar, mul_tensor_3, mul_tensor_4)

            tl.store(out_base_ptr + offsets, value, mask=mask)
            acc += tl.sum(tl.where(mask, value, 0.0), axis=0)

        tl.store(out_sum_ptr + cols, acc, mask=col_mask)


def _validate_inputs(inputs):
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for oracle_deeprecommender_selu_sum_permute.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    arg18_1, arg17_1, shape_param = inputs
    for name, value in (("arg18_1", arg18_1), ("arg17_1", arg17_1)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
        if value.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensor inputs")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
        if tuple(value.shape) != OUT_BASE_SHAPE:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {OUT_BASE_SHAPE}")
        if tuple(value.stride()) != OUT_BASE_STRIDE:
            raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {OUT_BASE_STRIDE}")
        if value.storage_offset() != 0:
            raise ValueError(f"{name} has storage_offset {value.storage_offset()}, expected 0")
    if arg18_1.device != arg17_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if list(shape_param) != [COLS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    return arg18_1, arg17_1


@oracle_impl(hardware="H100", shapes="(T([1024, 197951], f32), T([1024, 197951], f32), S([197951]))")
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
    arg18_1, arg17_1 = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        OUT_BASE_SHAPE,
        OUT_BASE_STRIDE,
        device=arg18_1.device,
        dtype=torch.float32,
    )
    out_sum = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=arg18_1.device,
        dtype=torch.float32,
    )

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_N"]),)
    _deeprecommender_selu_sum_permute_kernel[grid](
        arg18_1,
        arg17_1,
        out_base,
        out_sum,
        M=ROWS,
        N=COLS,
    )
    return out_base.permute(1, 0), out_sum


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
