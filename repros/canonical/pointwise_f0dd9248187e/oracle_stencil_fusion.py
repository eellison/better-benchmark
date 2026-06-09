"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete 2x2 stride-2 avg_pool2d -> BN-inference affine -> fp16 cast -> ReLU scope in one Triton output-tiled stencil kernel, reading each input window and channel affine vector once and storing only the final fp16 output, whereas Inductor currently materializes the fp16 pooled activation before the downstream broadcast pointwise chain; Inductor cannot do this today because scheduler fusion does not sink fixed-window pooling producers into their per-channel affine/ReLU consumers while preserving the fp16 pool and output cast boundaries; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit avg-pool stencil loops with fused broadcast affine epilogues for inference normalization patterns."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


EPS = 1.0e-5
BLOCK_C = 8
BLOCK_O = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _avgpool_bn_relu_kernel(
        x_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        OH: tl.constexpr,
        OW: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_O_: tl.constexpr,
        eps: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channels = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_linear = tl.program_id(2) * BLOCK_O_ + tl.arange(0, BLOCK_O_)

        oh = out_linear // OW
        ow = out_linear - oh * OW
        channel_mask = channels < C
        out_mask = out_linear < (OH * OW)
        mask = channel_mask[:, None] & out_mask[None, :]

        input_base = (
            batch * C * H * W
            + channels[:, None] * H * W
            + (oh[None, :] * 2) * W
            + ow[None, :] * 2
        )

        x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)
        x10 = tl.load(x_ptr + input_base + W, mask=mask, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + input_base + W + 1, mask=mask, other=0.0).to(tl.float32)

        # avg_pool2d returns fp16 here; round before the fp32 affine chain.
        pooled = ((x00 + x01 + x10 + x11) * 0.25).to(tl.float16).to(tl.float32)

        mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channels, mask=channel_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

        inv_std = 1.0 / tl.sqrt(var + eps)
        y = (pooled - mean[:, None]) * inv_std[:, None]
        y = y * weight[:, None] + bias[:, None]
        y = _relu_preserve_nan(y.to(tl.float16))

        output_offsets = batch * C * OH * OW + channels[:, None] * OH * OW + out_linear[None, :]
        tl.store(out_ptr + output_offsets, y, mask=mask)


def _require_f16_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    return value


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_87 must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"convolution_87 must be 4D, got shape {tuple(x.shape)}")
    if x.dtype != torch.float16:
        raise TypeError(f"convolution_87 has dtype {x.dtype}, expected torch.float16")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    if batch != 64:
        raise ValueError(f"unexpected batch size: {batch}")
    if height != width or height % 2 != 0:
        raise ValueError(f"expected even square spatial shape, got {(height, width)}")
    expected_x_stride = (channels * height * width, height * width, width, 1)
    if tuple(x.stride()) != expected_x_stride:
        raise ValueError(f"convolution_87 has stride {tuple(x.stride())}, expected {expected_x_stride}")

    vector_shape = (channels,)
    vector_stride = (1,)
    mean = _require_f16_tensor("arg441_1", inputs[1], vector_shape, vector_stride)
    var = _require_f16_tensor("arg442_1", inputs[2], vector_shape, vector_stride)
    weight = _require_f16_tensor("arg443_1", inputs[3], vector_shape, vector_stride)
    bias = _require_f16_tensor("arg444_1", inputs[4], vector_shape, vector_stride)

    if any(tensor.device != x.device for tensor in (mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, mean, var, weight, bias


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    x, mean, var, weight, bias = inputs
    pooled = torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2])
    y = (pooled - mean.float()[None, :, None, None]) * torch.reciprocal(
        torch.sqrt(var.float()[None, :, None, None] + EPS)
    )
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    return torch.relu(y.half())


@oracle_impl(hardware="H100", shapes="(T([64, 512, 14, 14], f16), T([512], f16), T([512], f16), T([512], f16), T([512], f16))")
def oracle_forward(inputs):
    """Run the full avg_pool2d, BN-inference affine, fp16 cast, and ReLU scope."""
    checked = _validate_inputs(inputs)
    x, mean, var, weight, bias = checked
    if not x.is_cuda:
        return _torch_oracle(checked)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    out_height = height // 2
    out_width = width // 2
    output = torch.empty_strided(
        (batch, channels, out_height, out_width),
        (channels * out_height * out_width, out_height * out_width, out_width, 1),
        device=x.device,
        dtype=torch.float16,
    )
    grid = (
        batch,
        triton.cdiv(channels, BLOCK_C),
        triton.cdiv(out_height * out_width, BLOCK_O),
    )
    _avgpool_bn_relu_kernel[grid](
        x,
        mean,
        var,
        weight,
        bias,
        output,
        C=channels,
        H=height,
        W=width,
        OH=out_height,
        OW=out_width,
        BLOCK_C_=BLOCK_C,
        BLOCK_O_=BLOCK_O,
        eps=EPS,
        num_warps=4,
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
        if any(isinstance(item, torch.Tensor) and item.is_cuda for item in inputs):
            torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False

    ok = True
    if eager.shape != oracle_out.shape:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        ok = False
    if eager.dtype != oracle_out.dtype:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        ok = False
    if eager.stride() != oracle_out.stride():
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        ok = False
    if not ok:
        return False

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~eager_nan
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
        ok = _check_oracle_nan_equal(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
