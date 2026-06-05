"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Repro.forward scope with coalesced channels-last BN-training stats, fused normalize-affine-ReLU6-spatial-mean output, and in-place running-stat updates, whereas Inductor's compiled lowering is already within measurement noise of this full-scope Triton implementation; Inductor cannot materially beat it today because the default shape is dominated by reading the 32 MB activation for statistics and the required pooled normalized output work rather than by an avoidable fusion boundary; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template case unless future codegen can reduce the required activation traffic or combine the stats and pooled-output passes without losing parallelism."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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

EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _partial_stats_tiled_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sq_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        elems_per_channel: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r_block = tl.program_id(0)
        c_block = tl.program_id(1) * BLOCK_C
        r = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
        c = c_block + tl.arange(0, BLOCK_C)
        r_mask = r < elems_per_channel
        c_mask = c < channels

        hw = height * width
        n = r // hw
        rem = r - n * hw
        h = rem // width
        w = rem - h * width
        offsets = (
            n[:, None] * x_stride_n
            + c[None, :] * x_stride_c
            + h[:, None] * x_stride_h
            + w[:, None] * x_stride_w
        )
        mask = r_mask[:, None] & c_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        partial_sum = tl.sum(x, axis=0)
        partial_sq = tl.sum(x * x, axis=0)
        out_offsets = r_block * channels + c
        tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_mask)
        tl.store(partial_sq_ptr + out_offsets, partial_sq, mask=c_mask)

    @triton.jit
    def _finalize_stats_tiled_kernel(
        partial_sum_ptr,
        partial_sq_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        inv_std_ptr,
        channels: tl.constexpr,
        n_blocks: tl.constexpr,
        elems_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_block = tl.program_id(0) * BLOCK_C
        b = tl.arange(0, BLOCK_B)
        c = c_block + tl.arange(0, BLOCK_C)
        mask = (b[:, None] < n_blocks) & (c[None, :] < channels)
        offsets = b[:, None] * channels + c[None, :]

        sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sqs = tl.load(partial_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = elems_per_channel + 0.0
        mean = tl.sum(sums, axis=0) / total
        mean_sq = tl.sum(sqs, axis=0) / total
        var = tl.maximum(mean_sq - mean * mean, 0.0)
        inv_std = tl.rsqrt(var + eps)

        c_mask = c < channels
        old_mean = tl.load(running_mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        new_mean = old_mean * (1.0 - momentum) + mean * momentum
        new_var = old_var * (1.0 - momentum) + var * running_var_correction * momentum

        tl.store(mean_ptr + c, mean, mask=c_mask)
        tl.store(inv_std_ptr + c, inv_std, mask=c_mask)
        tl.store(running_mean_ptr + c, new_mean, mask=c_mask)
        tl.store(running_var_ptr + c, new_var, mask=c_mask)

    @triton.jit
    def _bn_relu6_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        inv_std_ptr,
        out_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_block = tl.program_id(1) * BLOCK_C
        c = c_block + tl.arange(0, BLOCK_C)
        hw = tl.arange(0, BLOCK_HW)
        c_mask = c < channels
        hw_mask = hw < height * width

        h = hw // width
        w = hw - h * width
        offsets = (
            batch * x_stride_n
            + c[None, :] * x_stride_c
            + h[:, None] * x_stride_h
            + w[:, None] * x_stride_w
        )
        mask = hw_mask[:, None] & c_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        inv_std = tl.load(inv_std_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        y = (x - mean[None, :]) * inv_std[None, :]
        y = y * weight[None, :] + bias[None, :]
        y = tl.minimum(tl.maximum(y, 0.0), 6.0)
        pooled = tl.sum(tl.where(mask, y, 0.0), axis=0) / ((height * width) + 0.0)

        out_offsets = batch * channels + c
        tl.store(out_ptr + out_offsets, pooled, mask=c_mask)


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    cloned: list[Any] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            cloned.append(value.detach().clone())
        else:
            cloned.append(value)
    return cloned


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    x, running_mean, running_var, weight, bias, out_shape = inputs
    tensor_inputs = (
        ("convolution_51", x),
        ("primals_310", running_mean),
        ("primals_311", running_var),
        ("primals_312", weight),
        ("primals_313", bias),
    )
    for name, tensor in tensor_inputs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected tensor input {name}, got {type(tensor)!r}")
        if tensor.dtype is not torch.float32:
            raise TypeError(f"expected float32 {name}, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    if x.ndim != 4:
        raise ValueError(f"expected 4D input, got shape={tuple(x.shape)}")
    n_batches, channels, height, width = (int(dim) for dim in x.shape)
    if tuple(running_mean.shape) != (channels,) or tuple(running_var.shape) != (channels,):
        raise ValueError(
            f"running-stat shape mismatch: mean={tuple(running_mean.shape)} "
            f"var={tuple(running_var.shape)} channels={channels}"
        )
    if tuple(weight.shape) != (channels,) or tuple(bias.shape) != (channels,):
        raise ValueError(f"affine shape mismatch: weight={tuple(weight.shape)} bias={tuple(bias.shape)}")
    if tuple(out_shape) != (n_batches, channels):
        raise ValueError(f"unexpected reshape parameter {out_shape!r}, expected {(n_batches, channels)}")
    if height * width > 128:
        raise ValueError(f"oracle expects small spatial pooling tiles, got H*W={height * width}")

    return x, running_mean, running_var, weight, bias, (n_batches, channels, height, width)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full BN-training, ReLU6, spatial-pool, and running-stat update scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, running_mean, running_var, weight, bias, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    elems_per_channel = n_batches * height * width
    stats_block_r = 512
    stats_block_c = 16
    n_stat_blocks = triton.cdiv(elems_per_channel, stats_block_r)

    partial_sum = torch.empty((n_stat_blocks, channels), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty_like(partial_sum)
    mean = torch.empty((channels,), device=x.device, dtype=torch.float32)
    inv_std = torch.empty((channels,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided((n_batches, channels), (channels, 1), device=x.device, dtype=torch.float32)

    _partial_stats_tiled_kernel[(n_stat_blocks, triton.cdiv(channels, stats_block_c))](
        x,
        partial_sum,
        partial_sq,
        channels=channels,
        height=height,
        width=width,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        elems_per_channel=elems_per_channel,
        BLOCK_R=stats_block_r,
        BLOCK_C=stats_block_c,
        num_warps=8,
    )
    _finalize_stats_tiled_kernel[(triton.cdiv(channels, stats_block_c),)](
        partial_sum,
        partial_sq,
        running_mean,
        running_var,
        mean,
        inv_std,
        channels=channels,
        n_blocks=n_stat_blocks,
        elems_per_channel=elems_per_channel,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_B=triton.next_power_of_2(n_stat_blocks),
        BLOCK_C=stats_block_c,
        num_warps=1,
    )

    output_block_c = 64
    grid = (n_batches, triton.cdiv(channels, output_block_c))
    _bn_relu6_spatial_mean_kernel[grid](
        x,
        weight,
        bias,
        mean,
        inv_std,
        out,
        channels=channels,
        height=height,
        width=width,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        BLOCK_C=output_block_c,
        BLOCK_HW=triton.next_power_of_2(height * width),
        num_warps=2,
    )
    return out, running_mean, running_var


def _normalize_outputs(outputs: Any) -> list[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    return list(outputs)


def _check_oracle_fresh_inputs(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)
    with torch.no_grad():
        expected = instance(*eager_inputs)
        actual = oracle_forward(oracle_inputs)
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, eager produces {len(expected_list)}")
        return False

    all_ok = True
    for idx, (eager, oracle) in enumerate(zip(expected_list, actual_list)):
        shape_ok = eager.shape == oracle.shape
        stride_ok = eager.stride() == oracle.stride()
        dtype_ok = eager.dtype == oracle.dtype
        if not shape_ok:
            print(f"  output {idx}: SCOPE_MISMATCH shape oracle={list(oracle.shape)} eager={list(eager.shape)}")
            all_ok = False
            continue
        if not dtype_ok:
            print(f"  output {idx}: WARNING dtype mismatch oracle={oracle.dtype} eager={eager.dtype}")

        max_diff = (eager.float() - oracle.float()).abs().max().item()
        value_ok = torch.allclose(eager.float(), oracle.float(), atol=atol, rtol=rtol)
        ok = value_ok and stride_ok
        all_ok = all_ok and ok
        print(
            f"  output {idx}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(eager.shape)} dtype={eager.dtype} max_diff={max_diff:.2e} "
            f"expected_stride={eager.stride()} oracle_stride={oracle.stride()})"
        )
    return all_ok


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
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Accepted for template CLI compatibility; this repro has no stochastic outputs.",
    )
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
        ok = _check_oracle_fresh_inputs(instance, inputs, atol=args.atol, rtol=args.rtol)
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
