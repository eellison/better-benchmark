"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin residual add, channel LayerNorm affine, and 7x7 spatial mean by keeping only per-row normalization statistics and accumulating directly into the returned [128,1024] tensor, whereas Inductor currently schedules the LayerNorm pointwise affine and following spatial mean through generic reduction/pointwise work over the expanded [128,7,7,1024] activation; Inductor cannot do this today because its scheduler/codegen cannot sink a normalization epilogue through a second small spatial reduction while preserving the row-wise var_mean producer; the fix is SCHEDULER_FUSION: add a guarded LayerNorm-to-spatial-mean fusion template that stores row stats, fuses the affine epilogue into the downstream mean, and writes the final output layout directly."""
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True

BATCH = 128
SPATIAL = 49
HEIGHT = 7
WIDTH = 7
CHANNELS = 1024
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
STATS_BLOCK_C = 1024
MEAN_INV_SPATIAL_BLOCK = 64


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_layernorm_stats_kernel(
        addmm_ptr,
        residual_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C)
        mask = c_offsets < channels
        offsets = row * channels + c_offsets

        values = (
            tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.sum(values, axis=0) / channels
        centered = values - mean
        var = tl.sum(centered * centered, axis=0) / channels
        invstd = tl.rsqrt(var + eps)

        tl.store(mean_ptr + row, mean)
        tl.store(invstd_ptr + row, invstd)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 64}, num_warps=8, num_stages=3),
        ],
        key=["channels"],
    )
    @triton.jit
    def _layernorm_spatial_mean_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        s_offsets = tl.arange(0, BLOCK_S)
        c = c_offsets[:, None]
        s = s_offsets[None, :]
        row_offsets = batch * spatial + s
        mask = (c < channels) & (s < spatial)
        input_offsets = row_offsets * channels + c

        values = (
            tl.load(addmm_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.load(mean_ptr + row_offsets, mask=s < spatial, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + row_offsets, mask=s < spatial, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)

        normalized = (values - mean) * invstd
        affine = normalized * weight[:, None] + bias[:, None]
        affine = tl.where(s < spatial, affine, 0.0)
        pooled = tl.sum(affine, axis=1) / spatial

        tl.store(out_ptr + batch * channels + c_offsets, pooled, mask=c_offsets < channels)


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    addmm_95, view_652, arg361_1, arg362_1, shape0, shape1 = inputs
    addmm = _expect_f32_tensor("addmm_95", addmm_95, (ROWS, CHANNELS), (CHANNELS, 1))
    residual = _expect_f32_tensor(
        "view_652",
        view_652,
        (BATCH, SPATIAL, CHANNELS),
        (SPATIAL * CHANNELS, CHANNELS, 1),
    )
    weight = _expect_f32_tensor("arg361_1", arg361_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg362_1", arg362_1, (CHANNELS,), (1,))
    if list(shape0) != [BATCH, SPATIAL, CHANNELS]:
        raise ValueError(f"unexpected first reshape parameter: {shape0!r}")
    if list(shape1) != [BATCH, HEIGHT, WIDTH, CHANNELS]:
        raise ValueError(f"unexpected second reshape parameter: {shape1!r}")

    device = addmm.device
    if any(t.device != device for t in (residual, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return addmm, residual, weight, bias, shape0, shape1


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, shape0, shape1 = _validate_inputs(inputs)
    x = torch.reshape(addmm, shape0)
    x = torch.add(residual, x)
    x = torch.reshape(x, shape1)
    var, mean = torch.var_mean(x, dim=[3], correction=0, keepdim=True)
    x = torch.sub(x, mean)
    x = torch.mul(x, torch.rsqrt(torch.add(var, EPS)))
    x = torch.mul(x, weight)
    x = torch.add(x, bias)
    return torch.mean(x, dim=[1, 2])


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([128, 49, 1024], f32), T([1024], f32), T([1024], f32), S([128, 49, 1024]), S([128, 7, 7, 1024]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the same single output."""
    addmm, residual, weight, bias, _shape0, _shape1 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_reference(inputs)

    mean = torch.empty_strided((ROWS,), (1,), device=addmm.device, dtype=torch.float32)
    invstd = torch.empty_strided((ROWS,), (1,), device=addmm.device, dtype=torch.float32)
    out = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=addmm.device, dtype=torch.float32)

    _row_layernorm_stats_kernel[(ROWS,)](
        addmm,
        residual,
        mean,
        invstd,
        channels=CHANNELS,
        eps=EPS,
        BLOCK_C=STATS_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _layernorm_spatial_mean_kernel[
        lambda meta: (BATCH, triton.cdiv(CHANNELS, meta["BLOCK_C"]))
    ](
        addmm,
        residual,
        weight,
        bias,
        mean,
        invstd,
        out,
        channels=CHANNELS,
        spatial=SPATIAL,
        BLOCK_S=MEAN_INV_SPATIAL_BLOCK,
    )
    return out


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
        help="Disable auto-detection and skipping of stochastic outputs",
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
            true_floor = result["status"] == "GOOD"
            print(f"classification: {CLASSIFICATION}")
            print(f"true_floor: {'yes' if true_floor else 'no'} ({result['status']})")
            print(f"actionable: {'yes' if ACTIONABLE and true_floor else 'no'}")
            if not true_floor:
                print("diagnosis_only: not_true_floor because compiled Inductor is at least as fast as this full-scope oracle")


if __name__ == "__main__":
    main()
