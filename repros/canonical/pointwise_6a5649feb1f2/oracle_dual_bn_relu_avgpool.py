"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full dual BN-inference affine, fp16 branch rounding, fp16 branch sum, ReLU, and 2x2 stride-2 avg_pool2d scope in one output-tiled Triton stencil kernel that writes only the final fp16 `[32,1024,7,7]` tensor, whereas Inductor lowers the captured graph through generic pointwise producers plus a structured pooling consumer over the full `[32,1024,14,14]` activation; Inductor cannot do this today because scheduler fusion does not sink this broadcast-affine multi-producer ReLU epilogue into the fixed-window avg_pool2d stencil and generate the pooled output directly; the fix is SCHEDULER_FUSION: fuse broadcast pointwise producers into fixed-window pooling consumers and emit a single stencil-plus-epilogue loop nest."""
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


# --- Oracle kernel(s) ---
BATCH = 32
CHANNELS = 1024
HEIGHT = 14
WIDTH = 14
OUT_HEIGHT = 7
OUT_WIDTH = 7
OUT_HW = OUT_HEIGHT * OUT_WIDTH
EPS = 1.0e-5
BLOCK_C = 8
BLOCK_O = 64

if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _dual_bn_relu_avgpool_kernel(
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        x2_ptr,
        var2_ptr,
        weight2_ptr,
        bias2_ptr,
        out_ptr,
        BLOCK_C_: tl.constexpr,
        BLOCK_O_: tl.constexpr,
        eps: tl.constexpr,
    ):
        b = tl.program_id(0)
        c = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        o = tl.arange(0, BLOCK_O_)

        c_mask = c < 1024
        o_mask = o < 49
        mask = c_mask[:, None] & o_mask[None, :]

        oh = o // 7
        ow = o - oh * 7
        input_base = (
            b * 1024 * 196
            + c[:, None] * 196
            + (oh[None, :] * 2) * 14
            + ow[None, :] * 2
        )
        output_base = b * 1024 * 49 + c[:, None] * 49 + o[None, :]

        mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + c, mask=c_mask, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        var2 = tl.load(var2_ptr + c, mask=c_mask, other=1.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        inv1 = 1.0 / tl.sqrt(var1 + eps)
        inv2 = 1.0 / tl.sqrt(var2 + eps)
        acc = tl.zeros((BLOCK_C_, BLOCK_O_), dtype=tl.float32)

        for dh in tl.static_range(2):
            for dw in tl.static_range(2):
                pos = input_base + dh * 14 + dw
                x1 = tl.load(x1_ptr + pos, mask=mask, other=0.0).to(tl.float32)
                x2 = tl.load(x2_ptr + pos, mask=mask, other=0.0).to(tl.float32)

                y1 = (x1 - mean1[:, None]) * inv1[:, None]
                y1 = y1 * weight1[:, None] + bias1[:, None]
                y2 = (x2 - mean2[:, None]) * inv2[:, None]
                y2 = y2 * weight2[:, None] + bias2[:, None]

                # The repro rounds each branch to fp16 before the fp16 add.
                summed = (y1.to(tl.float16) + y2.to(tl.float16)).to(tl.float16)
                acc += _relu_preserve_nan(summed).to(tl.float32)

        tl.store(out_ptr + output_base, (acc * 0.25).to(tl.float16), mask=mask)


def _require_f16_tensor(name, value, shape, stride):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    mean1, x1, var1, weight1, bias1, mean2, x2, var2, weight2, bias2 = inputs
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    image_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    image_stride = (CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1)

    mean1 = _require_f16_tensor("arg91_1", mean1, vector_shape, vector_stride)
    x1 = _require_f16_tensor("convolution_19", x1, image_shape, image_stride)
    var1 = _require_f16_tensor("arg92_1", var1, vector_shape, vector_stride)
    weight1 = _require_f16_tensor("arg93_1", weight1, vector_shape, vector_stride)
    bias1 = _require_f16_tensor("arg94_1", bias1, vector_shape, vector_stride)
    mean2 = _require_f16_tensor("arg96_1", mean2, vector_shape, vector_stride)
    x2 = _require_f16_tensor("convolution_20", x2, image_shape, image_stride)
    var2 = _require_f16_tensor("arg97_1", var2, vector_shape, vector_stride)
    weight2 = _require_f16_tensor("arg98_1", weight2, vector_shape, vector_stride)
    bias2 = _require_f16_tensor("arg99_1", bias2, vector_shape, vector_stride)

    tensors = (mean1, x1, var1, weight1, bias1, mean2, x2, var2, weight2, bias2)
    if any(t.device != x1.device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(inputs):
    mean1, x1, var1, weight1, bias1, mean2, x2, var2, weight2, bias2 = inputs
    y1 = (x1 - mean1[None, :, None, None].float()) * torch.rsqrt(
        var1[None, :, None, None].float() + EPS
    )
    y1 = (y1 * weight1[None, :, None, None] + bias1[None, :, None, None]).half()
    y2 = (x2 - mean2[None, :, None, None].float()) * torch.rsqrt(
        var2[None, :, None, None].float() + EPS
    )
    y2 = (y2 * weight2[None, :, None, None] + bias2[None, :, None, None]).half()
    y = torch.relu((y1 + y2).half())
    return torch.ops.aten.avg_pool2d.default(y, [2, 2], [2, 2], [0, 0], True, False)


@oracle_impl(hardware="H100", shapes="(T([1024], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([1024], f16), T([32, 1024, 14, 14], f16), T([1024], f16), T([1024], f16), T([1024], f16))")
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
    tensors = _validate_inputs(inputs)
    if not tensors[1].is_cuda:
        return _torch_oracle(tensors)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    out = torch.empty_strided(
        (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1),
        device=tensors[1].device,
        dtype=torch.float16,
    )
    grid = (BATCH, triton.cdiv(CHANNELS, BLOCK_C))
    _dual_bn_relu_avgpool_kernel[grid](
        *tensors,
        out,
        BLOCK_C_=BLOCK_C,
        BLOCK_O_=BLOCK_O,
        eps=EPS,
        num_warps=4,
        num_stages=3,
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
