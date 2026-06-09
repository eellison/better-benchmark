"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `sum_sum_sum_ccb3a106dd79` NFNet backward fragment by fusing the avg_pool2d_backward expansion, both sigmoid pointwise chains, and both returned channel reductions into one Triton producer pass with two accumulators plus a tiny partial-sum finalizer, whereas Inductor currently keeps the spatial reduction feeding `sigmoid(arg176) * (1 - sigmoid(arg176))` as a materialized dependent reduction and schedules the sibling channel sum separately; Inductor cannot do this today because its algebraic simplifier/reduction codegen does not flatten linear `sum([2,3]) -> broadcast pointwise -> sum([0,2,3])` chains into the same multi-output reduction as a sibling sum; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to reassociate such linear dependent reductions and emit one multi-accumulator channel-reduction template over the shared fused producer."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


N = 128
C = 256
H = 56
W = 56
HW = H * W
POOL_H = 28
POOL_W = 28
POOL_HW = POOL_H * POOL_W
REDUCE_SIZE = N * HW
BLOCK_R = 4096
N_TILES = (REDUCE_SIZE + BLOCK_R - 1) // BLOCK_R
FINAL_BLOCK = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_reduce_kernel(
        getitem_207,
        getitem_204,
        arg176_1,
        arg173_1,
        arg158_1,
        partial_gate,
        partial_main,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        W_: tl.constexpr,
        POOL_W_: tl.constexpr,
        POOL_HW_: tl.constexpr,
        REDUCE_SIZE_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        r = tile * BLOCK + tl.arange(0, BLOCK)
        mask = r < REDUCE_SIZE_

        n = r // HW_
        hw = r - n * HW_
        h = hw // W_
        w = hw - h * W_
        pool_hw = (h // 2) * POOL_W_ + (w // 2)

        base56 = n * (C_ * HW_) + c * HW_ + hw
        base28 = n * (C_ * POOL_HW_) + c * POOL_HW_ + pool_hw
        gate_offset = n * C_ + c

        pool_grad = tl.load(getitem_207 + base28, mask=mask, other=0.0).to(tl.float32) * 0.25
        x204 = tl.load(getitem_204 + base56, mask=mask, other=0.0).to(tl.float32)
        gate_in = tl.load(arg176_1 + gate_offset, mask=mask, other=0.0).to(tl.float32)
        x173 = tl.load(arg173_1 + base56, mask=mask, other=0.0).to(tl.float32)
        x158 = tl.load(arg158_1 + base56, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = (x204 + pool_grad) * 0.9805806756909201
        gate = 1.0 / (tl.exp(-gate_in) + 1.0)
        gate_deriv = gate * (1.0 - gate)

        gated = x173 * gate
        gated = gated * 2.0
        gated = gated * 0.2
        activation_in = gated + x158

        activation = 1.0 / (tl.exp(-activation_in) + 1.0)
        main = mul_tensor * activation
        main = main * (activation_in * (1.0 - activation) + 1.0)

        gate_contrib = main * 0.2
        gate_contrib = gate_contrib * 2.0
        gate_contrib = gate_contrib * x173
        gate_contrib = gate_contrib * gate_deriv

        main = tl.where(mask, main, 0.0)
        gate_contrib = tl.where(mask, gate_contrib, 0.0)

        partial_offset = c * N_TILES_ + tile
        tl.store(partial_gate + partial_offset, tl.sum(gate_contrib, axis=0))
        tl.store(partial_main + partial_offset, tl.sum(main, axis=0))

    @triton.jit
    def _finalize_kernel(
        partial_gate,
        partial_main,
        out_gate,
        out_main,
        N_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N_TILES_
        base = c * N_TILES_ + offsets
        gate_vals = tl.load(partial_gate + base, mask=mask, other=0.0).to(tl.float32)
        main_vals = tl.load(partial_main + base, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_gate + c, tl.sum(gate_vals, axis=0))
        tl.store(out_main + c, tl.sum(main_vals, axis=0))


@oracle_impl(hardware="H100", shapes="(T([128, 256, 28, 28], f32), T([128, 256, 56, 56], f32), T([128, 256, 56, 56], f32), T([128, 256, 1, 1], f32), T([128, 256, 56, 56], f32), T([128, 256, 56, 56], f32))")
def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        getitem_207,
        _arg177_1,
        getitem_204,
        arg176_1,
        arg173_1,
        arg158_1,
    ) = inputs

    partial_gate = torch.empty((C, N_TILES), device=getitem_207.device, dtype=torch.float32)
    partial_main = torch.empty((C, N_TILES), device=getitem_207.device, dtype=torch.float32)
    out_gate = torch.empty((C,), device=getitem_207.device, dtype=torch.float32)
    out_main = torch.empty((C,), device=getitem_207.device, dtype=torch.float32)

    _partial_reduce_kernel[(C, N_TILES)](
        getitem_207,
        getitem_204,
        arg176_1,
        arg173_1,
        arg158_1,
        partial_gate,
        partial_main,
        C_=C,
        HW_=HW,
        W_=W,
        POOL_W_=POOL_W,
        POOL_HW_=POOL_HW,
        REDUCE_SIZE_=REDUCE_SIZE,
        N_TILES_=N_TILES,
        BLOCK=BLOCK_R,
        num_warps=8,
    )
    _finalize_kernel[(C,)](
        partial_gate,
        partial_main,
        out_gate,
        out_main,
        N_TILES_=N_TILES,
        BLOCK=FINAL_BLOCK,
        num_warps=8,
    )
    return (out_gate, out_main)


def main():
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


if __name__ == "__main__":
    main()
