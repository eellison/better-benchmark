"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ShuffleNet inference f16 graph `convert mean -> unsqueeze -> sub, convert var -> add 1e-5 -> sqrt -> reciprocal -> mul by 1, broadcast mul, broadcast f16 weight mul, broadcast f16 bias add, f16 cast, ReLU, cat([getitem_26, relu], dim=1), view [N,2,C,H,W], permute [0,2,1,3,4], contiguous clone, view [N,2*C,H,W]` in one Triton kernel that stores the final contiguous f16 output with stride `(2*C*H*W,H*W,W,1)`, even output channels loaded from the strided `getitem_26` input and odd output channels computed from BN/ReLU, whereas Inductor currently materializes the BN/ReLU branch and channel cat before a separate layout-only shuffle clone; Inductor cannot do this today because scheduler fusion does not push the consumer reshape/permute/clone indexing back through a mixed virtual-cat producer and per-channel affine epilogue; the fix is SCHEDULER_FUSION: let fixed channel-cat producers and their pointwise branches store directly in the final channel-shuffled consumer layout."""
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
EPS = 1.0e-5
BLOCK_N = 256

if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def oracle_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        getitem_ptr,
        output_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        HW: tl.constexpr,
        OUT_C: tl.constexpr,
        GETITEM_STRIDE_N: tl.constexpr,
        GETITEM_STRIDE_C: tl.constexpr,
        GETITEM_STRIDE_H: tl.constexpr,
        GETITEM_STRIDE_W: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        hw = offsets % HW
        out_c = (offsets // HW) % OUT_C
        batch = offsets // (OUT_C * HW)
        src_c = out_c // 2
        is_bn_relu = (out_c - src_c * 2) != 0

        h = hw // W
        w = hw - h * W
        src_offset = batch * (C * HW) + src_c * HW + hw
        getitem_offset = (
            batch * GETITEM_STRIDE_N
            + src_c * GETITEM_STRIDE_C
            + h * GETITEM_STRIDE_H
            + w * GETITEM_STRIDE_W
        )

        getitem_value = tl.load(
            getitem_ptr + getitem_offset,
            mask=mask & ~is_bn_relu,
            other=0.0,
        )

        conv_value = tl.load(
            conv_ptr + src_offset,
            mask=mask & is_bn_relu,
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(mean_ptr + src_c, mask=mask & is_bn_relu, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + src_c, mask=mask & is_bn_relu, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + src_c, mask=mask & is_bn_relu, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + src_c, mask=mask & is_bn_relu, other=0.0).to(tl.float32)

        y = (conv_value - mean) * (1.0 / tl.sqrt(var + 0.00001))
        y = y * weight + bias
        y = _relu_preserve_nan(y)
        out = tl.where(is_bn_relu, y, getitem_value)

        tl.store(output_ptr + offsets, out, mask=mask)


def _require_f16_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: tuple[object, ...] | list[object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    mean, conv, var, weight, bias, getitem, shape0, shape1 = inputs
    if not isinstance(conv, torch.Tensor) or conv.ndim != 4:
        raise TypeError("convolution_54 must be a rank-4 tensor")

    batch, channels, height, width = (int(dim) for dim in conv.shape)
    hw = height * width
    conv_stride = (channels * hw, hw, width, 1)
    getitem_stride = (2 * channels * hw, hw, width, 1)

    mean = _require_f16_tensor("arg272_1", mean, (channels,), (1,))
    conv = _require_f16_tensor("convolution_54", conv, (batch, channels, height, width), conv_stride)
    var = _require_f16_tensor("arg273_1", var, (channels,), (1,))
    weight = _require_f16_tensor("arg274_1", weight, (channels,), (1,))
    bias = _require_f16_tensor("arg275_1", bias, (channels,), (1,))
    getitem = _require_f16_tensor(
        "getitem_26",
        getitem,
        (batch, channels, height, width),
        getitem_stride,
    )

    expected_shape0 = (batch, 2, channels, height, width)
    expected_shape1 = (batch, 2 * channels, height, width)
    if tuple(shape0) != expected_shape0:
        raise ValueError(f"_shape_param_0 is {shape0}, expected {expected_shape0}")
    if tuple(shape1) != expected_shape1:
        raise ValueError(f"_shape_param_1 is {shape1}, expected {expected_shape1}")

    device = conv.device
    for tensor in (mean, var, weight, bias, getitem):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")

    return mean, conv, var, weight, bias, getitem, expected_shape0, expected_shape1


def _torch_oracle(
    mean: torch.Tensor,
    conv: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    getitem: torch.Tensor,
    shape0: tuple[int, ...],
    shape1: tuple[int, ...],
) -> torch.Tensor:
    y = conv - mean.float()[None, :, None, None]
    y = y * torch.reciprocal(torch.sqrt(var.float() + EPS))[None, :, None, None]
    y = y * weight[:, None, None] + bias[:, None, None]
    y = torch.relu(y.to(torch.float16))
    cat = torch.cat([getitem, y], dim=1)
    return cat.view(shape0).permute(0, 2, 1, 3, 4).clone(memory_format=torch.contiguous_format).view(shape1)


@oracle_impl(hardware="H100", shapes="(T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), T([512, 232, 7, 7], f16, stride=(22736, 49, 7, 1)), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))")
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
    mean, conv, var, weight, bias, getitem, shape0, shape1 = _validate_inputs(inputs)
    if not conv.is_cuda:
        return _torch_oracle(mean, conv, var, weight, bias, getitem, shape0, shape1)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    batch, channels, height, width = (int(dim) for dim in conv.shape)
    hw = height * width
    out_channels = 2 * channels
    out = torch.empty_strided(
        shape1,
        (out_channels * hw, hw, width, 1),
        device=conv.device,
        dtype=torch.float16,
    )
    n_elements = batch * out_channels * hw
    grid = (triton.cdiv(n_elements, BLOCK_N),)
    oracle_kernel[grid](
        mean,
        conv,
        var,
        weight,
        bias,
        getitem,
        out,
        N=n_elements,
        C=channels,
        H=height,
        W=width,
        HW=hw,
        OUT_C=out_channels,
        GETITEM_STRIDE_N=getitem.stride(0),
        GETITEM_STRIDE_C=getitem.stride(1),
        GETITEM_STRIDE_H=getitem.stride(2),
        GETITEM_STRIDE_W=getitem.stride(3),
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=4,
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
