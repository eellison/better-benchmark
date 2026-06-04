"""Oracle for var_mean_e48f702d6f81.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
groupnorm-like Repro.forward scope in one Triton kernel, keeping each
16-channel group in registers while producing the squeezed mean [64,32],
squeezed rsqrt [64,32], and affine ReLU [64,512,1,1] outputs, whereas Inductor
currently lowers this through generic var_mean/norm scheduling rather than a
dedicated small-group GroupNorm forward template that can materialize auxiliary
stats and the normalized activation from the same register tile; Inductor
cannot do this today because the norm template canonicalizer does not recognize
this multi-output GroupNorm+affine+ReLU pattern with exposed mean/rsqrt
outputs; the fix is NEW_PATTERN: add a canonical GroupNorm forward pattern that
emits stats plus normalized affine/ReLU in one fused reduction-pointwise
template.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (
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

BATCH = 64
CHANNELS = 512
GROUPS = 32
GROUP_SIZE = 16
EPS = 1.0e-5

INPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
STATS_SHAPE = (BATCH, GROUPS)
RELU_SHAPE = (BATCH, CHANNELS, 1, 1)
RELU_STRIDE = (CHANNELS, 1, 1, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_GROUPS": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 4}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_GROUPS": 32}, num_warps=4, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _groupnorm_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_out_ptr,
        rsqrt_out_ptr,
        relu_out_ptr,
        total_groups: tl.constexpr,
        groups: tl.constexpr,
        channels_total: tl.constexpr,
        group_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        group_rows = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        channels = tl.arange(0, BLOCK_C)
        valid_rows = group_rows < total_groups

        batch = group_rows // groups
        group = group_rows - batch * groups
        channel = group[:, None] * group_size + channels[None, :]

        offsets = batch[:, None] * channels_total + channel
        mask = valid_rows[:, None] & (channels[None, :] < group_size)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / group_size
        centered = x - mean[:, None]
        var = tl.sum(centered * centered, axis=1) / group_size
        inv_std = tl.rsqrt(var + eps)

        w = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        relu = tl.maximum(centered * inv_std[:, None] * w + b, 0.0)

        stats_offsets = batch * groups + group
        tl.store(mean_out_ptr + stats_offsets, mean, mask=valid_rows)
        tl.store(rsqrt_out_ptr + stats_offsets, inv_std, mask=valid_rows)
        tl.store(relu_out_ptr + offsets, relu, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, weight, bias, group_shape, relu_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"expected tensor input 1, got {type(weight)!r}")
    if not isinstance(bias, torch.Tensor):
        raise TypeError(f"expected tensor input 2, got {type(bias)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(weight.shape) != (CHANNELS,):
        raise ValueError(f"unexpected weight shape: {tuple(weight.shape)}")
    if tuple(bias.shape) != (CHANNELS,):
        raise ValueError(f"unexpected bias shape: {tuple(bias.shape)}")
    if tuple(group_shape) != (BATCH, GROUPS, GROUP_SIZE, 1):
        raise ValueError(f"unexpected group view shape parameter: {group_shape!r}")
    if tuple(relu_shape) != RELU_SHAPE:
        raise ValueError(f"unexpected relu view shape parameter: {relu_shape!r}")

    for name, tensor in (("x", x), ("weight", weight), ("bias", bias)):
        if tensor.dtype is not torch.float32:
            raise ValueError(f"unexpected {name} dtype: {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError("oracle_groupnorm_relu.py expects CUDA inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"expected contiguous {name}, got stride={tensor.stride()}")

    return x, weight, bias


def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward groupnorm + affine + ReLU computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_groupnorm_relu.py")

    x, weight, bias = _validate_inputs(inputs)
    mean_out = torch.empty(STATS_SHAPE, device=x.device, dtype=torch.float32)
    rsqrt_out = torch.empty(STATS_SHAPE, device=x.device, dtype=torch.float32)
    relu_out = torch.empty_strided(RELU_SHAPE, RELU_STRIDE, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(BATCH * GROUPS, meta["BLOCK_GROUPS"]),)
    _groupnorm_relu_kernel[grid](
        x,
        weight,
        bias,
        mean_out,
        rsqrt_out,
        relu_out,
        total_groups=BATCH * GROUPS,
        groups=GROUPS,
        channels_total=CHANNELS,
        group_size=GROUP_SIZE,
        eps=EPS,
        BLOCK_C=GROUP_SIZE,
    )
    return mean_out, rsqrt_out, relu_out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_list = list(expected)
    actual_list = list(actual)
    all_ok = True
    for i, (eager, oracle) in enumerate(zip(expected_list, actual_list)):
        ok = eager.stride() == oracle.stride()
        all_ok = all_ok and ok
        print(
            f"  output {i} layout: {'PASS' if ok else 'FAIL'} "
            f"(expected_stride={eager.stride()}, oracle_stride={oracle.stride()})"
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
        ok = ok and _check_layout(instance, inputs)
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
