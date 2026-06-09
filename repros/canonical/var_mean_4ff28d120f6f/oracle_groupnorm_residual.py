"""Oracle for var_mean_4ff28d120f6f.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
GroupNorm-like Repro.forward scope in one Triton kernel, reducing each
float32 [64,32,8,4] group tile, writing the squeezed mean [64,32], squeezed
rsqrt [64,32], and affine residual ReLU [64,256,2,2] outputs directly, whereas
Inductor lowers the same graph through its generic multi-output var_mean/norm
schedule; Inductor cannot do this today because the norm template canonicalizer
does not recognize exposed-stat GroupNorm forward with per-channel affine,
residual add, and ReLU as a dedicated small-group template; the fix is
NEW_PATTERN: add a canonical GroupNorm+residual forward lowering that emits
stats plus normalized affine/ReLU in one shape-specialized reduction-pointwise
kernel.
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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

BATCH = 64
CHANNELS = 256
GROUPS = 32
CHANNELS_PER_GROUP = 8
HEIGHT = 2
WIDTH = 2
HW = HEIGHT * WIDTH
GROUP_ELEMS = CHANNELS_PER_GROUP * HW
EPS = 1.0e-5

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
GROUP_VIEW_SHAPE = (BATCH, GROUPS, CHANNELS_PER_GROUP, HW)
STATS_SHAPE = (BATCH, GROUPS)
FINAL_SHAPE = INPUT_SHAPE
FINAL_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
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
    def _groupnorm_residual_kernel(
        x_ptr,
        scale_ptr,
        bias_ptr,
        residual_ptr,
        mean_out_ptr,
        rsqrt_out_ptr,
        final_out_ptr,
        total_groups: tl.constexpr,
        groups: tl.constexpr,
        channels_total: tl.constexpr,
        channels_per_group: tl.constexpr,
        hw: tl.constexpr,
        group_elems: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        group_rows = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
        elems = tl.arange(0, BLOCK_ELEMS)
        valid_rows = group_rows < total_groups
        valid_elems = elems < group_elems

        batch = group_rows // groups
        group = group_rows - batch * groups
        channel_in_group = elems // hw
        channel = group[:, None] * channels_per_group + channel_in_group[None, :]

        offsets = batch[:, None] * channels_total * hw + group[:, None] * group_elems + elems[None, :]
        mask = valid_rows[:, None] & valid_elems[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / group_elems
        centered = x - mean[:, None]
        variance = tl.sum(tl.where(valid_elems[None, :], centered * centered, 0.0), axis=1) / group_elems
        inv_std = tl.rsqrt(variance + eps)

        scale = tl.load(scale_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        final = tl.maximum(centered * inv_std[:, None] * scale + bias + residual, 0.0)

        stats_offsets = batch * groups + group
        tl.store(mean_out_ptr + stats_offsets, mean, mask=valid_rows)
        tl.store(rsqrt_out_ptr + stats_offsets, inv_std, mask=valid_rows)
        tl.store(final_out_ptr + offsets, final, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    x, scale, bias, residual, group_shape, final_shape = inputs
    for index, value in enumerate((x, scale, bias, residual)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected tensor input {index}, got {type(value)!r}")

    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected convolution shape: {tuple(x.shape)}")
    if tuple(scale.shape) != (CHANNELS,):
        raise ValueError(f"unexpected scale shape: {tuple(scale.shape)}")
    if tuple(bias.shape) != (CHANNELS,):
        raise ValueError(f"unexpected bias shape: {tuple(bias.shape)}")
    if tuple(residual.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected residual shape: {tuple(residual.shape)}")
    if tuple(group_shape) != GROUP_VIEW_SHAPE:
        raise ValueError(f"unexpected group view shape parameter: {group_shape!r}")
    if tuple(final_shape) != FINAL_SHAPE:
        raise ValueError(f"unexpected final view shape parameter: {final_shape!r}")

    for name, tensor in (("convolution", x), ("scale", scale), ("bias", bias), ("residual", residual)):
        if tensor.dtype is not torch.float32:
            raise ValueError(f"unexpected {name} dtype: {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError("oracle_groupnorm_residual.py expects CUDA inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"expected contiguous {name}, got stride={tensor.stride()}")

    return x, scale, bias, residual


@oracle_impl(hardware="H100", shapes="(T([64, 256, 2, 2], f32), T([256], f32), T([256], f32), T([64, 256, 2, 2], f32), S([64, 32, 8, 4]), S([64, 256, 2, 2]))")
def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward GroupNorm + residual ReLU computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same three outputs: squeezed mean, squeezed rsqrt, and final ReLU tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_groupnorm_residual.py")

    x, scale, bias, residual = _validate_inputs(inputs)
    mean_out = torch.empty(STATS_SHAPE, device=x.device, dtype=torch.float32)
    rsqrt_out = torch.empty(STATS_SHAPE, device=x.device, dtype=torch.float32)
    final_out = torch.empty_strided(FINAL_SHAPE, FINAL_STRIDE, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(BATCH * GROUPS, meta["BLOCK_GROUPS"]),)
    _groupnorm_residual_kernel[grid](
        x,
        scale,
        bias,
        residual,
        mean_out,
        rsqrt_out,
        final_out,
        total_groups=BATCH * GROUPS,
        groups=GROUPS,
        channels_total=CHANNELS,
        channels_per_group=CHANNELS_PER_GROUP,
        hw=HW,
        group_elems=GROUP_ELEMS,
        eps=EPS,
        BLOCK_ELEMS=triton.next_power_of_2(GROUP_ELEMS),
    )
    return mean_out, rsqrt_out, final_out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    all_ok = True
    for i, (eager, oracle) in enumerate(zip(expected, actual)):
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
