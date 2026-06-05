"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Repro.forward four-branch BN-inference affine, NaN-preserving ReLU, and channel concat by writing each branch directly into its final [128,768,17,17] channel slice, whereas Inductor currently materializes the four decomposed broadcast pointwise branch results and then runs a separate concat/layout kernel; Inductor cannot do this today because its scheduler does not model aten.cat as an output-layout epilogue for sibling pointwise producers with fixed channel partitions; the fix is SCHEDULER_FUSION: teach the scheduler to fuse fixed-shape concat outputs by assigning each producer a destination channel interval in one generated pointwise kernel."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
BRANCH_C = 192
NBRANCH = 4
OUT_C = BRANCH_C * NBRANCH
HEIGHT = 17
WIDTH = 17
HW = HEIGHT * WIDTH
BRANCH_NUMEL = BATCH * BRANCH_C * HW
OUT_SHAPE = (BATCH, OUT_C, HEIGHT, WIDTH)
OUT_STRIDE = (OUT_C * HW, HW, WIDTH, 1)
EPS = 0.001
BLOCK_SIZE = 256


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _cat_bn_relu_kernel(
        conv0,
        mean0,
        var0,
        weight0,
        bias0,
        conv1,
        mean1,
        var1,
        weight1,
        bias1,
        conv2,
        mean2,
        var2,
        weight2,
        bias2,
        conv3,
        mean3,
        var3,
        weight3,
        bias3,
        out,
        BLOCK_SIZE_: tl.constexpr,
    ):
        branch = tl.program_id(1)
        offsets = tl.program_id(0) * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        mask = offsets < 7102464

        hw = offsets % 289
        channel = (offsets // 289) % 192
        batch = offsets // 55488
        input_offsets = (batch * 192 + channel) * 289 + hw
        output_offsets = (batch * 768 + branch * 192 + channel) * 289 + hw

        is0 = branch == 0
        is1 = branch == 1
        is2 = branch == 2
        is3 = branch == 3

        x = tl.load(conv0 + input_offsets, mask=mask & is0, other=0.0).to(tl.float32)
        x += tl.load(conv1 + input_offsets, mask=mask & is1, other=0.0).to(tl.float32)
        x += tl.load(conv2 + input_offsets, mask=mask & is2, other=0.0).to(tl.float32)
        x += tl.load(conv3 + input_offsets, mask=mask & is3, other=0.0).to(tl.float32)

        mean = tl.load(mean0 + channel, mask=mask & is0, other=0.0).to(tl.float32)
        mean += tl.load(mean1 + channel, mask=mask & is1, other=0.0).to(tl.float32)
        mean += tl.load(mean2 + channel, mask=mask & is2, other=0.0).to(tl.float32)
        mean += tl.load(mean3 + channel, mask=mask & is3, other=0.0).to(tl.float32)

        var = tl.load(var0 + channel, mask=mask & is0, other=0.0).to(tl.float32)
        var += tl.load(var1 + channel, mask=mask & is1, other=0.0).to(tl.float32)
        var += tl.load(var2 + channel, mask=mask & is2, other=0.0).to(tl.float32)
        var += tl.load(var3 + channel, mask=mask & is3, other=0.0).to(tl.float32)

        weight = tl.load(weight0 + channel, mask=mask & is0, other=0.0).to(tl.float32)
        weight += tl.load(weight1 + channel, mask=mask & is1, other=0.0).to(tl.float32)
        weight += tl.load(weight2 + channel, mask=mask & is2, other=0.0).to(tl.float32)
        weight += tl.load(weight3 + channel, mask=mask & is3, other=0.0).to(tl.float32)

        bias = tl.load(bias0 + channel, mask=mask & is0, other=0.0).to(tl.float32)
        bias += tl.load(bias1 + channel, mask=mask & is1, other=0.0).to(tl.float32)
        bias += tl.load(bias2 + channel, mask=mask & is2, other=0.0).to(tl.float32)
        bias += tl.load(bias3 + channel, mask=mask & is3, other=0.0).to(tl.float32)

        y = (x - mean) * (1.0 / tl.sqrt(var + 0.001))
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out + output_offsets, y, mask=mask)


def _require_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 20:
        raise ValueError(f"{REPRO_ID} expects 20 inputs, got {len(inputs)}")

    vector_shape = (BRANCH_C,)
    image_shape = (BATCH, BRANCH_C, HEIGHT, WIDTH)
    image_stride = (BRANCH_C * HW, HW, WIDTH, 1)
    vector_stride = (1,)

    tensors = (
        _require_tensor("arg302_1", inputs[0], vector_shape, vector_stride),
        _require_tensor("convolution_60", inputs[1], image_shape, image_stride),
        _require_tensor("arg303_1", inputs[2], vector_shape, vector_stride),
        _require_tensor("arg304_1", inputs[3], vector_shape, vector_stride),
        _require_tensor("arg305_1", inputs[4], vector_shape, vector_stride),
        _require_tensor("arg317_1", inputs[5], vector_shape, vector_stride),
        _require_tensor("convolution_63", inputs[6], image_shape, image_stride),
        _require_tensor("arg318_1", inputs[7], vector_shape, vector_stride),
        _require_tensor("arg319_1", inputs[8], vector_shape, vector_stride),
        _require_tensor("arg320_1", inputs[9], vector_shape, vector_stride),
        _require_tensor("arg342_1", inputs[10], vector_shape, vector_stride),
        _require_tensor("convolution_68", inputs[11], image_shape, image_stride),
        _require_tensor("arg343_1", inputs[12], vector_shape, vector_stride),
        _require_tensor("arg344_1", inputs[13], vector_shape, vector_stride),
        _require_tensor("arg345_1", inputs[14], vector_shape, vector_stride),
        _require_tensor("arg347_1", inputs[15], vector_shape, vector_stride),
        _require_tensor("convolution_69", inputs[16], image_shape, image_stride),
        _require_tensor("arg348_1", inputs[17], vector_shape, vector_stride),
        _require_tensor("arg349_1", inputs[18], vector_shape, vector_stride),
        _require_tensor("arg350_1", inputs[19], vector_shape, vector_stride),
    )

    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return tensors


def oracle_forward(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_relu.py")

    (
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
        mean2,
        conv2,
        var2,
        weight2,
        bias2,
        mean3,
        conv3,
        var3,
        weight3,
        bias3,
    ) = _validate_inputs(inputs)

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=conv0.device, dtype=torch.float32)
    grid = (triton.cdiv(BRANCH_NUMEL, BLOCK_SIZE), NBRANCH)
    _cat_bn_relu_kernel[grid](
        conv0,
        mean0,
        var0,
        weight0,
        bias0,
        conv1,
        mean1,
        var1,
        weight1,
        bias1,
        conv2,
        mean2,
        var2,
        weight2,
        bias2,
        conv3,
        mean3,
        var3,
        weight3,
        bias3,
        out,
        BLOCK_SIZE_=BLOCK_SIZE,
        num_warps=4,
        num_stages=4,
    )
    return out


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
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False
    if tuple(oracle_out.shape) != OUT_SHAPE or tuple(eager.shape) != OUT_SHAPE:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        return False
    if oracle_out.dtype is not torch.float32 or eager.dtype is not torch.float32:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        return False
    if tuple(oracle_out.stride()) != OUT_STRIDE or tuple(eager.stride()) != OUT_STRIDE:
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        return False

    eager_nan = torch.isnan(eager)
    oracle_nan = torch.isnan(oracle_out)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~eager_nan
    if finite.any():
        max_diff = (eager[finite] - oracle_out[finite]).abs().max().item()
        values_ok = torch.allclose(eager[finite], oracle_out[finite], atol=atol, rtol=rtol)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
