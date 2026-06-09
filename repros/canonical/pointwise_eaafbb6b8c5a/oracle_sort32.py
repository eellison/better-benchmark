"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full `aten.sort.default` int64[32] value/index return in one Triton bitonic kernel while preserving ATen's unstable equal-key ordering and contiguous output layout, whereas Inductor currently routes this tiny sort through its generic ATen-backed sort lowering; Inductor cannot do this today because scheduler/codegen has no fixed-size small integer value/index sort template for N=32; the fix is NEW_PATTERN: add a guarded small-N integer sort lowering that emits compact value/index bitonic code directly."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "NEW_PATTERN"
OUT_SHAPE = (32,)
OUT_STRIDE = (1,)


from oracle_harness import (
    oracle_impl,  # noqa: E402
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


if triton is not None:

    @triton.jit
    def _sort32_i64_kernel(input_ptr, values_ptr, indices_ptr, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        vals = tl.load(input_ptr + offsets)
        idxs = offsets.to(tl.int64)

        # Mirror ATen's bitonicSortKVInPlace schedule for n <= 32. The default
        # CUDA sort is explicitly unstable: equal keys swap when the substage's
        # ascending compare/swap direction asks for it.
        for size in tl.static_range(2, 32, 2):
            if size == 2 or size == 4 or size == 8 or size == 16:
                for stride in tl.static_range(8, 0, -1):
                    if (
                        (stride == 8 or stride == 4 or stride == 2 or stride == 1)
                        and stride <= size // 2
                    ):
                        partners = offsets ^ stride
                        other_vals = tl.gather(vals, partners, 0)
                        other_idxs = tl.gather(idxs, partners, 0)
                        low_lane = offsets < partners
                        lower_offsets = tl.minimum(offsets, partners)
                        thread_idx = (lower_offsets // (2 * stride)) * stride + (
                            lower_offsets & (stride - 1)
                        )
                        descending_pair = (thread_idx & (size // 2)) != 0

                        lower_vals = tl.where(low_lane, vals, other_vals)
                        lower_idxs = tl.where(low_lane, idxs, other_idxs)
                        upper_vals = tl.where(low_lane, other_vals, vals)
                        upper_idxs = tl.where(low_lane, other_idxs, idxs)

                        swap = (lower_vals < upper_vals) == descending_pair
                        new_lower_vals = tl.where(swap, upper_vals, lower_vals)
                        new_lower_idxs = tl.where(swap, upper_idxs, lower_idxs)
                        new_upper_vals = tl.where(swap, lower_vals, upper_vals)
                        new_upper_idxs = tl.where(swap, lower_idxs, upper_idxs)
                        vals = tl.where(low_lane, new_lower_vals, new_upper_vals)
                        idxs = tl.where(low_lane, new_lower_idxs, new_upper_idxs)

        for stride in tl.static_range(16, 0, -1):
            if stride == 16 or stride == 8 or stride == 4 or stride == 2 or stride == 1:
                partners = offsets ^ stride
                other_vals = tl.gather(vals, partners, 0)
                other_idxs = tl.gather(idxs, partners, 0)
                low_lane = offsets < partners

                lower_vals = tl.where(low_lane, vals, other_vals)
                lower_idxs = tl.where(low_lane, idxs, other_idxs)
                upper_vals = tl.where(low_lane, other_vals, vals)
                upper_idxs = tl.where(low_lane, other_idxs, idxs)

                swap = lower_vals >= upper_vals
                new_lower_vals = tl.where(swap, upper_vals, lower_vals)
                new_lower_idxs = tl.where(swap, upper_idxs, lower_idxs)
                new_upper_vals = tl.where(swap, lower_vals, upper_vals)
                new_upper_idxs = tl.where(swap, lower_idxs, upper_idxs)
                vals = tl.where(low_lane, new_lower_vals, new_upper_vals)
                idxs = tl.where(low_lane, new_lower_idxs, new_upper_idxs)

        tl.store(values_ptr + offsets, vals)
        tl.store(indices_ptr + offsets, idxs)


@oracle_impl(hardware="H100", shapes="(T([32], i64))")
def oracle_forward(inputs):
    """Run the full-scope sort oracle.

    SCOPE INVARIANT: accepts the same single input as Repro.forward() and
    returns `(sorted_values, sorted_indices)` with the same shapes and dtypes as
    `torch.ops.aten.sort.default(arg0_1)`.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sort32.py")

    (arg0_1,) = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg0_1)!r}")
    if tuple(arg0_1.shape) != (32,):
        raise ValueError(f"unexpected input shape: {tuple(arg0_1.shape)}")
    if arg0_1.dtype is not torch.int64:
        raise ValueError(f"unexpected input dtype: {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError("oracle_sort32.py expects CUDA inputs")

    sorted_values = torch.empty_like(arg0_1)
    sorted_indices = torch.empty_like(arg0_1)
    _sort32_i64_kernel[(1,)](arg0_1, sorted_values, sorted_indices, BLOCK=32)
    return sorted_values, sorted_indices


def _as_tuple(outputs):
    if isinstance(outputs, tuple):
        return outputs
    return (outputs,)


def _report_exact_diff_and_layout(instance, inputs) -> bool:
    with torch.no_grad():
        eager_outputs = _as_tuple(instance(*inputs))
        oracle_outputs = _as_tuple(oracle_forward(inputs))
        torch.cuda.synchronize()

    ok = True
    for i, (eager, oracle) in enumerate(zip(eager_outputs, oracle_outputs)):
        diff = (eager - oracle).abs().max().item()
        layout_ok = tuple(oracle.shape) == OUT_SHAPE and oracle.stride() == OUT_STRIDE
        exact_ok = diff == 0
        ok = ok and exact_ok and layout_ok
        print(
            f"  output {i}: max_abs_diff={diff} "
            f"layout={'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(oracle.shape)} stride={oracle.stride()})"
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
    parser.add_argument("--rtol", type=float, default=0.0,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=0.0,
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
        ok = _report_exact_diff_and_layout(instance, inputs) and ok
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
                    print(
                        "WARNING: oracle is slower than compile for "
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
                print(
                    "WARNING: oracle is slower than compile "
                    f"(ratio={result['ratio']:.3f}x)"
                )


if __name__ == "__main__":
    main()
