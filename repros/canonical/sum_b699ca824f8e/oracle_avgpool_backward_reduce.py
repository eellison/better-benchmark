"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the adaptive-average-pool backward broadcast and SiLU-gradient channel sum as a direct per-channel reduction from the `[128, 2304]` pooled gradient and `[128, 2304, 7, 7]` activation tensor, whereas Inductor currently lowers the `full -> as_strided_scatter -> as_strided -> expand -> div -> pointwise -> sum([0, 2, 3])` chain as ordinary tensor producers and consumers; Inductor cannot do this today because its scatter/view scheduler does not recognize a zero-fill structured scatter/expand whose only real consumer is a channel reduction and cannot lower that pattern to an output-centric reduction; the fix is SCATTER_REDUCE: add an FX/post-grad rewrite for zero-fill structured scatter/expand feeding pointwise channel reductions and lower it to a direct channel-reduction template."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - allows CPU-only syntax checks.
    triton = None
    tl = None


REPRO_ID = "sum_b699ca824f8e"

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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
SHAPE_LABEL = "timm_nfnet_l0_train_369f714a"
SPATIAL_SIZE = 49.0


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_direct_avgpool_backward_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    """Direct reduction for the avgpool-backward expand feeding a channel sum."""
    sigmoid = torch.reciprocal(torch.exp(torch.neg(convolution_80)) + 1.0)
    silu_backward = sigmoid * (convolution_80 * (1.0 - sigmoid) + 1.0)
    spatial_reduced = silu_backward.sum(dim=(2, 3))
    return (mm * spatial_reduced).sum(dim=0) / SPATIAL_SIZE


if triton is not None:

    @triton.jit
    def _avgpool_backward_reduce_kernel(
        mm_ptr,
        conv_ptr,
        out_ptr,
        stride_conv_n: tl.constexpr,
        stride_conv_c: tl.constexpr,
        stride_conv_h: tl.constexpr,
        stride_conv_w: tl.constexpr,
        n_size: tl.constexpr,
        c_size: tl.constexpr,
        block_c: tl.constexpr,
        block_n: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_n = tl.program_id(1)
        c_offsets = pid_c * block_c + tl.arange(0, block_c)
        row_offsets = tl.arange(0, block_rows)
        n_offsets = pid_n * block_n + row_offsets // 64
        spatial_offsets = row_offsets % 64
        h_offsets = spatial_offsets // 7
        w_offsets = spatial_offsets - h_offsets * 7
        valid_spatial = spatial_offsets < 49
        mask = (
            (n_offsets[:, None] < n_size)
            & valid_spatial[:, None]
            & (c_offsets[None, :] < c_size)
        )
        conv_offsets = (
            n_offsets[:, None] * stride_conv_n
            + c_offsets[None, :] * stride_conv_c
            + h_offsets[:, None] * stride_conv_h
            + w_offsets[:, None] * stride_conv_w
        )
        mm_offsets = n_offsets[:, None] * c_size + c_offsets[None, :]
        x = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0)
        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0)
        sigmoid = 1.0 / (tl.exp(-x) + 1.0)
        vals = pooled_grad * sigmoid * (x * (1.0 - sigmoid) + 1.0) * (1.0 / 49.0)
        acc = tl.sum(vals, axis=0)
        tl.atomic_add(out_ptr + c_offsets, acc, sem="relaxed", mask=c_offsets < c_size)


def triton_direct_avgpool_backward_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda" or convolution_80.device.type != "cuda":
        raise RuntimeError("triton-direct requires CUDA inputs")

    out = torch.empty((mm.shape[1],), device=mm.device, dtype=mm.dtype)
    out.zero_()
    block_c = 16
    block_n = 8
    block_rows = block_n * 64
    grid = (triton.cdiv(mm.shape[1], block_c), triton.cdiv(mm.shape[0], block_n))
    _avgpool_backward_reduce_kernel[grid](
        mm,
        convolution_80,
        out,
        convolution_80.stride(0),
        convolution_80.stride(1),
        convolution_80.stride(2),
        convolution_80.stride(3),
        mm.shape[0],
        mm.shape[1],
        block_c,
        block_n,
        block_rows,
        num_warps=8,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([128, 2304], f32), T([128, 2304, 7, 7], f32, stride=(112896, 1, 16128, 2304)), S([128, 2304, 1, 1]), S([128, 2304, 7, 7]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return triton_direct_avgpool_backward_reduce(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
