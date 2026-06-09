"""
Oracle for sum_fa194db93388

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete `where(arg187 <= 0, full_1, getitem_210).sum([0, 2, 3])` in one
specialized Triton channel-reduction kernel for the fixed
`f32[128, 128, 1, 1] -> f32[128]` shape, whereas Inductor already emits one
fused `le/where/sum` reduction kernel for the same full scope; Inductor cannot
expose a distinct scheduler-fusion, scatter-reduce, cooperative split-K,
algebraic-elimination, or recompute-fusion opportunity here because the graph
is already a single producer-to-reduction region and the remaining cost is
tiny-kernel launch/reduction-template overhead; the fix is BANDWIDTH_BOUND:
only generic small-reduction launch/codegen tuning would help both paths, and
this full-scope oracle records the measured small-reduction floor when it beats
the required compile configs.
"""
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


N = 128
C = 128
H = 1
W = 1


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _where_sum_nchw_128_kernel(
        arg_ptr,
        full_ptr,
        getitem_ptr,
        out_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        n_offsets = tl.arange(0, N_)
        offsets = n_offsets[:, None] * C_ + c_offsets[None, :]
        mask = c_offsets[None, :] < C_

        lhs = tl.load(arg_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        rhs = tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        selected = tl.where(lhs <= 0.0, full_value, rhs)
        selected = tl.where(mask, selected, 0.0)
        totals = tl.sum(selected, axis=0)
        tl.store(out_ptr + c_offsets, totals, mask=c_offsets < C_)


def _validate_inputs(
    arg187_1: torch.Tensor,
    full_1: torch.Tensor,
    getitem_210: torch.Tensor,
) -> None:
    for name, tensor in (("arg187_1", arg187_1), ("getitem_210", getitem_210)):
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
    if not isinstance(full_1, torch.Tensor):
        raise TypeError(f"full_1 must be a tensor, got {type(full_1)!r}")
    if not full_1.is_cuda:
        raise ValueError("full_1 must be a CUDA tensor")
    if full_1.dtype is not torch.float32:
        raise ValueError(f"full_1 must be float32, got {full_1.dtype}")
    if tuple(full_1.shape) != ():
        raise ValueError(f"full_1 must be scalar-shaped, got {tuple(full_1.shape)}")


@oracle_impl(hardware="H100", shapes="(T([128, 128, 1, 1], f32), T([], f32), T([128, 128, 1, 1], f32))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    The repro consumes `(arg187_1, full_1, getitem_210)`, selects either the
    scalar `full_1` or `getitem_210` for every `arg187_1 <= 0` element in
    `f32[128, 128, 1, 1]`, then reduces over `[0, 2, 3]` to return one
    contiguous `f32[128]` tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_multi_output.py")
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    arg187_1, full_1, getitem_210 = inputs
    _validate_inputs(arg187_1, full_1, getitem_210)

    out = torch.empty_strided((C,), (1,), device=arg187_1.device, dtype=torch.float32)
    block_c = 8
    _where_sum_nchw_128_kernel[(triton.cdiv(C, block_c),)](
        arg187_1,
        full_1,
        getitem_210,
        out,
        N_=N,
        C_=C,
        BLOCK_C=block_c,
        num_warps=8,
    )
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
