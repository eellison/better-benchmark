"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Reformer virtual-cat LayerNorm scope in one shape-specialized Triton row kernel, including the fp16 residual add feeding the second cat half, fp32 var_mean over all 512 features, affine epilogue, and final fp16 store, whereas Inductor lowers the decomposed view/add/cat/var_mean/affine graph through a generic normalization schedule; Inductor cannot do this today because its scheduler does not recognize a fixed hidden-size concat producer with a partially computed half-row as a virtual input to a row-normalization template; the fix is SCHEDULER_FUSION: teach the normalization scheduler to inline static concat producers and their pointwise inputs into the row-reduction plan."""
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

ROWS = 4096
HALF_HIDDEN = 256
HIDDEN = 512
OUT_SHAPE = (1, ROWS, HIDDEN)
OUT_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
ADDMM_SHAPE = (ROWS, HALF_HIDDEN)
HALF_SHAPE = (1, ROWS, HALF_HIDDEN)
HALF_STRIDE = (ROWS * HALF_HIDDEN, HALF_HIDDEN, 1)
AFFINE_SHAPE = (HIDDEN,)
EPS = 1.0e-12

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
        ],
        key=["num_rows"],
    )
    @triton.jit
    def _virtual_cat_layernorm_kernel(
        addmm_ptr,
        add40_ptr,
        add47_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        num_rows: tl.constexpr,
        half_hidden: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        valid = rows[:, None] < num_rows

        left_mask = valid & (cols[None, :] < half_hidden)
        left = tl.load(
            add47_ptr + rows[:, None] * half_hidden + cols[None, :],
            mask=left_mask,
            other=0.0,
        ).to(tl.float32)

        right_cols = cols - half_hidden
        right_mask = valid & (cols[None, :] >= half_hidden)
        add40 = tl.load(
            add40_ptr + rows[:, None] * half_hidden + right_cols[None, :],
            mask=right_mask,
            other=0.0,
        )
        addmm = tl.load(
            addmm_ptr + rows[:, None] * half_hidden + right_cols[None, :],
            mask=right_mask,
            other=0.0,
        )
        right = (add40 + addmm).to(tl.float16).to(tl.float32)
        x = left + tl.where(cols[None, :] >= half_hidden, right, 0.0)

        mean = tl.sum(x, axis=1) / hidden
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / hidden
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        y = centered * invstd[:, None] * weight[None, :] + bias[None, :]
        tl.store(
            out_ptr + rows[:, None] * hidden + cols[None, :],
            y,
            mask=valid,
        )


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _require_shape_param(value: Any) -> tuple[int, int, int]:
    shape = tuple(int(dim) for dim in value)
    expected = (1, ROWS, HALF_HIDDEN)
    if shape != expected:
        raise ValueError(f"_shape_param_0 is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_11 = _require_tensor(
        "addmm_11",
        inputs[0],
        ADDMM_SHAPE,
        torch.float16,
        (HALF_HIDDEN, 1),
    )
    add_40 = _require_tensor("add_40", inputs[1], HALF_SHAPE, torch.float16, HALF_STRIDE)
    add_47 = _require_tensor("add_47", inputs[2], HALF_SHAPE, torch.float16, HALF_STRIDE)
    weight = _require_tensor("arg76_1", inputs[3], AFFINE_SHAPE, torch.float16, (1,))
    bias = _require_tensor("arg77_1", inputs[4], AFFINE_SHAPE, torch.float16, (1,))
    _require_shape_param(inputs[5])

    device = addmm_11.device
    for tensor in (add_40, add_47, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_11, add_40, add_47, weight, bias


@oracle_impl(hardware="H100", shapes="(T([4096, 256], f16), T([1, 4096, 256], f16), T([1, 4096, 256], f16), T([512], f16), T([512], f16), S([1, 4096, 256]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full Repro.forward virtual-cat LayerNorm computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fused_layernorm.py")

    addmm_11, add_40, add_47, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm_11.device,
        dtype=torch.float16,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _virtual_cat_layernorm_kernel[grid](
        addmm_11,
        add_40,
        add_47,
        weight,
        bias,
        output,
        num_rows=ROWS,
        half_hidden=HALF_HIDDEN,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=HIDDEN,
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
