"""
Oracle for var_mean_var_mean_d850ee31d68b

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Repro.forward scope in one shape-specialized Triton tile, reducing both independent `[N, group]` rows over 16 elements, storing the four returned mean/rsqrt stat tensors, and directly writing the summed affine ReLU output, whereas coordinate-descent Inductor already generates one full-scope kernel for the same tiny dual-groupnorm graph and the remaining differences are hand-specialized indexing versus generic norm scheduling; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute fusion because the cost is dominated by one launch plus the required f32 activation/affine reads and five output writes for a sub-megabyte workload; the fix is BANDWIDTH_BOUND: record this as an at-floor oracle and avoid opening a new Inductor optimization unless a backend change reduces single-kernel launch/allocation overhead.
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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


NUM_GROUPS = 32
EPS = 1.0e-5


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=3),
        ],
        key=["total_groups", "channels", "hw_size", "group_elems"],
    )
    @triton.jit
    def _dual_groupnorm_stats_relu_kernel(
        x1_ptr,
        weight1_ptr,
        bias1_ptr,
        x2_ptr,
        weight2_ptr,
        bias2_ptr,
        mean1_ptr,
        rstd1_ptr,
        mean2_ptr,
        rstd2_ptr,
        out_ptr,
        total_groups: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        group_elems: tl.constexpr,
        num_groups: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        elems = tl.arange(0, BLOCK_K)
        mask = (rows[:, None] < total_groups) & (elems[None, :] < group_elems)
        offsets = rows[:, None] * group_elems + elems[None, :]

        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        count = group_elems + 0.0
        mean1 = tl.sum(x1, axis=1) / count
        mean2 = tl.sum(x2, axis=1) / count

        centered1 = tl.where(mask, x1 - mean1[:, None], 0.0)
        centered2 = tl.where(mask, x2 - mean2[:, None], 0.0)
        var1 = tl.sum(centered1 * centered1, axis=1) / count
        var2 = tl.sum(centered2 * centered2, axis=1) / count
        rstd1 = tl.rsqrt(var1 + eps)
        rstd2 = tl.rsqrt(var2 + eps)

        row_mask = rows < total_groups
        tl.store(mean1_ptr + rows, mean1, mask=row_mask)
        tl.store(rstd1_ptr + rows, rstd1, mask=row_mask)
        tl.store(mean2_ptr + rows, mean2, mask=row_mask)
        tl.store(rstd2_ptr + rows, rstd2, mask=row_mask)

        channels_per_group: tl.constexpr = channels // num_groups
        group_id = rows % num_groups
        channel = group_id[:, None] * channels_per_group + elems[None, :] // hw_size

        scale1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        shift1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        scale2 = tl.load(weight2_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        shift2 = tl.load(bias2_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y1 = centered1 * rstd1[:, None] * scale1 + shift1
        y2 = centered2 * rstd2[:, None] * scale2 + shift2
        out = tl.maximum(y1 + y2, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects ten inputs, got {len(inputs)}")

    (
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        group_shape1,
        out_shape1,
        group_shape2,
        out_shape2,
    ) = inputs

    tensor_inputs = (x1, weight1, bias1, x2, weight2, bias2)
    if not all(isinstance(inp, torch.Tensor) for inp in tensor_inputs):
        raise TypeError("expected the first six inputs to be tensors")
    if x1.ndim != 4 or x2.ndim != 4:
        raise ValueError(f"expected 4D activation tensors, got {tuple(x1.shape)} and {tuple(x2.shape)}")
    if x1.dtype != torch.float32 or x2.dtype != torch.float32:
        raise TypeError(f"expected f32 activations, got {x1.dtype} and {x2.dtype}")
    if any(t.dtype != torch.float32 for t in (weight1, bias1, weight2, bias2)):
        raise TypeError(
            "expected f32 affine tensors, got "
            f"{weight1.dtype}, {bias1.dtype}, {weight2.dtype}, {bias2.dtype}"
        )
    if not all(t.is_cuda for t in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(t.is_contiguous() for t in tensor_inputs):
        raise ValueError(
            "expected contiguous tensors, got "
            f"x1_stride={x1.stride()} weight1_stride={weight1.stride()} "
            f"bias1_stride={bias1.stride()} x2_stride={x2.stride()} "
            f"weight2_stride={weight2.stride()} bias2_stride={bias2.stride()}"
        )
    if tuple(x1.shape) != tuple(x2.shape):
        raise ValueError(f"activation shapes differ: {tuple(x1.shape)} vs {tuple(x2.shape)}")

    n_batches, channels, height, width = (int(dim) for dim in x1.shape)
    if channels % NUM_GROUPS != 0:
        raise ValueError(f"channels must be divisible by {NUM_GROUPS}, got {channels}")
    for name, tensor in (
        ("weight1", weight1),
        ("bias1", bias1),
        ("weight2", weight2),
        ("bias2", bias2),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"expected {name} shape ({channels},), got {tuple(tensor.shape)}")

    hw_size = height * width
    expected_group_shape = (n_batches, NUM_GROUPS, channels // NUM_GROUPS, hw_size)
    expected_out_shape = tuple(x1.shape)
    for name, shape in (("group_shape1", group_shape1), ("group_shape2", group_shape2)):
        if not isinstance(shape, (list, tuple)) or tuple(shape) != expected_group_shape:
            raise ValueError(f"unexpected {name}: {shape!r}, expected {expected_group_shape}")
    for name, shape in (("out_shape1", out_shape1), ("out_shape2", out_shape2)):
        if not isinstance(shape, (list, tuple)) or tuple(shape) != expected_out_shape:
            raise ValueError(f"unexpected {name}: {shape!r}, expected {expected_out_shape}")

    return x1, weight1, bias1, x2, weight2, bias2, (n_batches, channels, height, width)


@oracle_impl(hardware="H100", shapes="(T([64, 512, 1, 1], f32), T([512], f32), T([512], f32), T([64, 512, 1, 1], f32), T([512], f32), T([512], f32), S([64, 32, 16, 1]), S([64, 512, 1, 1]), S([64, 32, 16, 1]), S([64, 512, 1, 1]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full dual-groupnorm stats plus summed ReLU computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same five outputs in order: mean1, rsqrt1, mean2, rsqrt2, relu.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x1, weight1, bias1, x2, weight2, bias2, shape = _validate_inputs(inputs)
    n_batches, channels, height, width = shape
    hw_size = height * width
    group_elems = channels // NUM_GROUPS * hw_size
    total_groups = n_batches * NUM_GROUPS
    block_k = triton.next_power_of_2(group_elems)

    stat_shape = (n_batches, NUM_GROUPS)
    stat_stride = _contiguous_stride(stat_shape)
    mean1 = torch.empty_strided(stat_shape, stat_stride, device=x1.device, dtype=x1.dtype)
    rstd1 = torch.empty_strided(stat_shape, stat_stride, device=x1.device, dtype=x1.dtype)
    mean2 = torch.empty_strided(stat_shape, stat_stride, device=x1.device, dtype=x1.dtype)
    rstd2 = torch.empty_strided(stat_shape, stat_stride, device=x1.device, dtype=x1.dtype)
    out = torch.empty_strided(tuple(x1.shape), _contiguous_stride(tuple(x1.shape)), device=x1.device, dtype=x1.dtype)

    grid = lambda meta: (triton.cdiv(total_groups, meta["BLOCK_M"]),)
    _dual_groupnorm_stats_relu_kernel[grid](
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        mean1,
        rstd1,
        mean2,
        rstd2,
        out,
        total_groups=total_groups,
        channels=channels,
        hw_size=hw_size,
        group_elems=group_elems,
        num_groups=NUM_GROUPS,
        eps=EPS,
        BLOCK_K=block_k,
    )
    return mean1, rstd1, mean2, rstd2, out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, tuple) or not isinstance(actual, tuple) or len(expected) != len(actual):
        print("  layout: FAIL (output structure mismatch)")
        return False

    ok = True
    for idx, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        output_ok = expected_tensor.stride() == actual_tensor.stride()
        ok = ok and output_ok
        print(
            f"  output {idx} layout: {'PASS' if output_ok else 'FAIL'} "
            f"(expected_stride={expected_tensor.stride()}, oracle_stride={actual_tensor.stride()})"
        )
    return ok


# --- CLI entry point ---
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
