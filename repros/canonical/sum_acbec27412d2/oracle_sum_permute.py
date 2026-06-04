"""
Oracle for sum_acbec27412d2.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full
permute/reshape/sum/reshape/permute return tuple by streaming the logical
`[128, 768]` source once, materializing the required strided `[768, 128]`
transpose output while accumulating the sibling `[768]` batch sum, whereas
Inductor lowers the reduction and the layout-return path as separate scheduled
concerns rather than one multi-output materializing reduction; Inductor cannot
do this today because its scheduler/codegen does not form a fused template that
shares loads between a reduction output and a same-source layout output with
different rank/stride contracts; the fix is SCHEDULER_FUSION: add support for
multi-output reductions that can fuse compatible view/permute materialization
stores with the shared reduction producer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
HEADS = 12
HEAD_DIM = 64
K = HEADS * HEAD_DIM
INPUT_SHAPE = (BATCH, HEADS, 1, HEAD_DIM)
INPUT_STRIDE = (K, HEAD_DIM, HEAD_DIM, 1)
OUT0_SHAPE = (K,)
OUT0_STRIDE = (1,)
OUT1_SHAPE = (K, BATCH)
OUT1_STRIDE = (1, K)
BLOCK_B = 128
BLOCK_K = 16
NUM_WARPS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _sum_and_permute_kernel(
        x_ptr,
        out_sum_ptr,
        out_transpose_ptr,
        BLOCK_B_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
        K_: tl.constexpr,
    ):
        k_offsets = tl.program_id(0) * BLOCK_K_ + tl.arange(0, BLOCK_K_)
        b_offsets = tl.arange(0, BLOCK_B_)
        mask = k_offsets[None, :] < K_
        source_offsets = b_offsets[:, None] * K_ + k_offsets[None, :]

        values = tl.load(x_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_transpose_ptr + source_offsets, values, mask=mask)
        sums = tl.sum(values, axis=0)
        tl.store(out_sum_ptr + k_offsets, sums, mask=k_offsets < K_)


def _require_tensor(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
    if tuple(tensor.shape) != INPUT_SHAPE:
        raise ValueError(f"{name} expected shape {INPUT_SHAPE}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} expected dtype torch.float32, got {tensor.dtype}")
    if tuple(tensor.stride()) != INPUT_STRIDE:
        raise ValueError(f"{name} expected stride {INPUT_STRIDE}, got {tuple(tensor.stride())}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return tensor


def _require_shape_param(value: object, name: str, expected: list[int]) -> None:
    if list(value) != expected:
        raise ValueError(f"{name} expected {expected}, got {value!r}")


def oracle_forward(inputs):
    """Run the full Repro.forward scope with one fused Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum_permute.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    getitem_142, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    x = _require_tensor(getitem_142, "getitem_142")
    _require_shape_param(_shape_param_0, "_shape_param_0", [BATCH, 1, K])
    _require_shape_param(_shape_param_1, "_shape_param_1", [K])
    _require_shape_param(_shape_param_2, "_shape_param_2", [BATCH, K])

    out_sum = torch.empty_strided(
        OUT0_SHAPE,
        OUT0_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    out_transpose = torch.empty_strided(
        OUT1_SHAPE,
        OUT1_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _sum_and_permute_kernel[(triton.cdiv(K, BLOCK_K),)](
        x,
        out_sum,
        out_transpose,
        BLOCK_B_=BLOCK_B,
        BLOCK_K_=BLOCK_K,
        K_=K,
        num_warps=NUM_WARPS,
    )
    return out_sum, out_transpose


def _check_layouts(outputs: object, expected_outputs: object) -> bool:
    if not isinstance(outputs, (tuple, list)) or not isinstance(expected_outputs, (tuple, list)):
        print("  layout: FAIL (expected tuple/list outputs)")
        return False
    if len(outputs) != len(expected_outputs):
        print(f"  layout: FAIL (oracle outputs={len(outputs)} eager outputs={len(expected_outputs)})")
        return False

    ok = True
    for idx, (got, expected) in enumerate(zip(outputs, expected_outputs)):
        layout_ok = (
            isinstance(got, torch.Tensor)
            and isinstance(expected, torch.Tensor)
            and tuple(got.shape) == tuple(expected.shape)
            and got.stride() == expected.stride()
            and got.dtype == expected.dtype
        )
        print(
            f"  output {idx} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(got.shape)} stride={got.stride()} dtype={got.dtype})"
        )
        ok = ok and layout_ok
    return ok


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
        with torch.no_grad():
            expected = instance(*inputs)
            actual = oracle_forward(inputs)
            torch.cuda.synchronize()
        ok = ok and _check_layouts(actual, expected)
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
