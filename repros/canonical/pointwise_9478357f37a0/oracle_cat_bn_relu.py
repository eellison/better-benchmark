"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full virtual channel cat, BN-inference affine in fp32 from fp16 parameters, fp16 cast, and ReLU in one Triton kernel, whereas Inductor currently materializes the fixed channel cat before the downstream pointwise BN/ReLU work; Inductor cannot do this today because its scheduler does not model aten.cat as a virtual multi-source producer that can be inlined into elementwise consumers with per-channel source selection; the fix is SCHEDULER_FUSION: allow fixed-shape channel concat producers to feed fused pointwise consumers directly instead of forcing a dense concatenated intermediate."""
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


EPS = 1.0e-5
TAIL_CHANNELS = 32
BLOCK_ROWS = 16
BLOCK_HW = 64


if triton is not None:

    @triton.jit
    def _cat_bn_relu_kernel(
        x0_ptr,
        x1_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C0: tl.constexpr,
        C1: tl.constexpr,
        CHANNELS: tl.constexpr,
        HW: tl.constexpr,
        TOTAL_ROWS: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.program_id(1) * BLOCK_HW_ + tl.arange(0, BLOCK_HW_)

        batch = row_offsets // CHANNELS
        channel = row_offsets - batch * CHANNELS
        valid_rows = row_offsets < TOTAL_ROWS
        valid_hw = hw_offsets < HW
        valid = valid_rows[:, None] & valid_hw[None, :]

        out_offsets = row_offsets[:, None] * HW + hw_offsets[None, :]

        x0_offsets = (batch[:, None] * C0 + channel[:, None]) * HW + hw_offsets[None, :]
        x1_channel = channel - C0
        x1_offsets = (batch[:, None] * C1 + x1_channel[:, None]) * HW + hw_offsets[None, :]
        use_x1 = channel >= C0
        src_ptrs = tl.where(use_x1[:, None], x1_ptr + x1_offsets, x0_ptr + x0_offsets)

        x = tl.load(src_ptrs, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=valid_rows, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * (1.0 / tl.sqrt(var[:, None] + EPSILON))
        y = y * weight[:, None] + bias[:, None]
        y_fp16 = y.to(tl.float16)
        relu = tl.where(y_fp16 < 0.0, 0.0, y_fp16)
        tl.store(out_ptr + out_offsets, relu, mask=valid)


def _require_tensor(
    name: str,
    value: object,
    *,
    ndim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape={tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype={dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[object] | tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_relu.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    x0 = _require_tensor("avg_pool2d_2", inputs[0], ndim=4, dtype=torch.float16)
    x1 = _require_tensor("convolution_89", inputs[1], ndim=4, dtype=torch.float16)
    mean = _require_tensor("arg451_1", inputs[2], ndim=1, dtype=torch.float16)
    var = _require_tensor("arg452_1", inputs[3], ndim=1, dtype=torch.float16)
    weight = _require_tensor("arg453_1", inputs[4], ndim=1, dtype=torch.float16)
    bias = _require_tensor("arg454_1", inputs[5], ndim=1, dtype=torch.float16)

    batch, c0, height, width = x0.shape
    if tuple(x1.shape) != (batch, TAIL_CHANNELS, height, width):
        raise ValueError(
            f"convolution_89 has shape={tuple(x1.shape)}, "
            f"expected={(batch, TAIL_CHANNELS, height, width)}"
        )

    channels = c0 + x1.shape[1]
    for name, tensor in (
        ("arg451_1", mean),
        ("arg452_1", var),
        ("arg453_1", weight),
        ("arg454_1", bias),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} has shape={tuple(tensor.shape)}, expected={(channels,)}")

    device = x0.device
    for tensor in (x1, mean, var, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return x0, x1, mean, var, weight, bias


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
    x0, x1, mean, var, weight, bias = _validate_inputs(inputs)
    batch, c0, height, width = x0.shape
    c1 = x1.shape[1]
    channels = c0 + c1
    hw = height * width

    output = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x0.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(batch * channels, BLOCK_ROWS), triton.cdiv(hw, BLOCK_HW))
    _cat_bn_relu_kernel[grid](
        x0,
        x1,
        mean,
        var,
        weight,
        bias,
        output,
        C0=c0,
        C1=c1,
        CHANNELS=channels,
        HW=hw,
        TOTAL_ROWS=batch * channels,
        EPSILON=EPS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    return output


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[object],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        if any(isinstance(inp, torch.Tensor) and inp.is_cuda for inp in inputs):
            torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False
    if tuple(oracle_out.shape) != tuple(eager.shape):
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        return False
    if oracle_out.dtype != eager.dtype:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        return False
    if tuple(oracle_out.stride()) != tuple(eager.stride()):
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        return False

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~(eager_nan | oracle_nan)
    if finite.any():
        max_diff = (eager_f32[finite] - oracle_f32[finite]).abs().max().item()
        values_ok = torch.allclose(eager_f32[finite], oracle_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(eager_nan.sum().item())})"
    )
    return ok


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
        ok = _check_oracle_nan_equal(instance, inputs, atol=args.atol, rtol=args.rtol)
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
