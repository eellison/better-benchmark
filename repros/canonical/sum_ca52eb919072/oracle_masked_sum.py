"""
Oracle for sum_ca52eb919072

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete `where(arg408_1 <= 0, 0.0, getitem_3).sum([0, 2, 3])` scope in one
shape-specialized Triton channel-reduction kernel for the fixed
`f32[128, 768, 1, 1] -> f32[768]` shape, whereas Inductor already emits one
fused mask/where/reduction kernel for the same full computation; Inductor
cannot expose a distinct scheduler-fusion, scatter-reduce, cooperative split-K,
algebraic-elimination, recompute-fusion, or new-pattern opportunity because the
graph is already a single producer-to-reduction region and the remaining cost
is the required two-input read, f32 accumulation, output store, and one GPU
launch; the fix is BANDWIDTH_BOUND: only generic small-reduction launch/codegen
tuning would help both paths, and this full-scope oracle records the measured
floor only if it beats the required tuned Inductor configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 768
H = 1
W = 1
OUTPUT_SHAPE = (C,)
OUTPUT_STRIDE = (1,)


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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 32}, num_warps=8, num_stages=3),
        ],
        key=["N_", "C_"],
    )
    @triton.jit
    def _masked_sum_nchw_128x768_kernel(
        arg_ptr,
        getitem_ptr,
        out_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = tl.arange(0, N_)
        mask = cols[None, :] < C_
        offsets = rows[:, None] * C_ + cols[None, :]

        arg = tl.load(arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        getitem = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        selected = tl.where(arg <= 0.0, 0.0, getitem)
        selected = tl.where(mask, selected, 0.0)
        totals = tl.sum(selected, axis=0)
        tl.store(out_ptr + cols, totals, mask=cols < C_)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg408_1, getitem_3 = inputs
    for name, tensor in (("arg408_1", arg408_1), ("getitem_3", getitem_3)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype is not torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")
        if tuple(tensor.shape) != (N, C, H, W):
            raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != (C, 1, 1, 1):
            raise ValueError(f"{name} has unexpected stride {tensor.stride()}")

    return arg408_1, getitem_3


@oracle_impl(hardware="H100", shapes="(T([128, 768, 1, 1], f32), T([128, 768, 1, 1], f32))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: accepts the same two tensors as Repro.forward(), applies
    the `arg408_1 <= 0` mask and zero-fill inside the timed Triton kernel, then
    reduces over `[0, 2, 3]` to return the single contiguous `f32[768]` output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_masked_sum.py")

    arg408_1, getitem_3 = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=arg408_1.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(C, meta["BLOCK_C"]),)
    _masked_sum_nchw_128x768_kernel[grid](arg408_1, getitem_3, out, N_=N, C_=C)
    return out


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
                    print(f"diagnosis_only: required comparison shows not_true_floor "
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
