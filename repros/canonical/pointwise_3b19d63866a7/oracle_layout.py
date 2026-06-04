"""
Full-scope oracle for pointwise_3b19d63866a7.

Gap diagnosis (classification: BANDWIDTH_BOUND): The oracle performs the exact `view -> select -> index -> view` computation by emitting one Triton row-gather kernel that writes the final contiguous `float32[32, 128, 1]` layout directly, whereas Inductor already has to perform the same small indexed load/store materialization for `aten.index.Tensor`; Inductor cannot remove the work because the dynamic index tensor may contain arbitrary valid rows and the advanced-index result is a new dense tensor, so the fixing classification is BANDWIDTH_BOUND: no scheduler or pattern change is expected to beat the launch plus 16 KiB gather/store floor except reducing generic launch/materialization overhead.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "BANDWIDTH_BOUND"


from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(int(size), 1)
    return tuple(reversed(strides))


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "N"],
    )
    @triton.jit
    def _row_gather_kernel(
        index_ptr,
        wait_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        N: tl.constexpr,
        INDEX_STRIDE0: tl.constexpr,
        WAIT_STRIDE0: tl.constexpr,
        WAIT_STRIDE1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        dst_row = offsets // N
        col = offsets - dst_row * N
        src_row = tl.load(index_ptr + dst_row * INDEX_STRIDE0, mask=mask, other=0)
        values = tl.load(
            wait_ptr + src_row * WAIT_STRIDE0 + col * WAIT_STRIDE1,
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs):
    """Run the full repro with one Triton gather kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 3:
        raise AssertionError(f"expected 3 inputs, got {len(inputs)}")

    arg268_1, wait_tensor, shape_param = inputs
    if not isinstance(arg268_1, torch.Tensor):
        raise TypeError(f"arg268_1 must be a tensor, got {type(arg268_1)!r}")
    if not isinstance(wait_tensor, torch.Tensor):
        raise TypeError(f"wait_tensor must be a tensor, got {type(wait_tensor)!r}")
    if arg268_1.dtype is not torch.int64:
        raise TypeError(f"arg268_1 must be int64, got {arg268_1.dtype}")
    if wait_tensor.dim() != 2:
        raise ValueError(f"wait_tensor must be rank 2, got shape={tuple(wait_tensor.shape)}")
    if arg268_1.dim() != 1:
        raise ValueError(f"arg268_1 must be rank 1, got shape={tuple(arg268_1.shape)}")
    if not arg268_1.is_cuda or not wait_tensor.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA tensor inputs")

    out_shape = tuple(int(dim) for dim in shape_param)
    total = int(arg268_1.numel() * wait_tensor.shape[1])
    if math.prod(out_shape) != total:
        raise ValueError(
            f"shape_param {out_shape} has {math.prod(out_shape)} elements, expected {total}"
        )

    out = torch.empty_strided(
        out_shape,
        _contiguous_strides(out_shape),
        device=wait_tensor.device,
        dtype=wait_tensor.dtype,
    )
    n_cols = int(wait_tensor.shape[1])
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK"]),)
    _row_gather_kernel[grid](
        arg268_1,
        wait_tensor,
        out,
        TOTAL=total,
        N=n_cols,
        INDEX_STRIDE0=int(arg268_1.stride(0)),
        WAIT_STRIDE0=int(wait_tensor.stride(0)),
        WAIT_STRIDE1=int(wait_tensor.stride(1)),
    )
    return out


def _check_layout(inputs) -> bool:
    with torch.no_grad():
        expected = get_repro_instance()(*inputs)
        actual = oracle_forward(inputs)
    if actual.is_cuda:
        torch.cuda.synchronize()

    ok = tuple(actual.shape) == tuple(expected.shape) and actual.stride() == expected.stride()
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()})"
    )
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
        ok = _check_layout(inputs) and ok
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
