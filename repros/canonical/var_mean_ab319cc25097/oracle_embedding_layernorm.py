"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot token embedding, positional embedding add, fp32 variance/mean normalization, affine scale/bias, and three returned `[2048,2560]` alias views in one Triton row-normalization kernel, whereas Inductor materializes the gathered embedding/add region before the norm template; Inductor cannot do this today when the normalization producer includes indexed embedding loads because the scheduler/norm canonicalization does not sink those row-local gathers and the positional add directly into the reduction loop; the fix is SCHEDULER_FUSION: allow rowwise gather/add producers with matching `[rows, hidden]` iteration space to be fused into the generated layernorm kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
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


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _embedding_layernorm_kernel(
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        H: tl.constexpr,
        POSITIONS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        token_id = tl.load(token_ids + row)
        pos = row % POSITIONS
        token = tl.load(token_table + token_id * H + offs, mask=mask, other=0.0).to(tl.float32)
        position = tl.load(position_table + pos * H + offs, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, token + position, 0.0)

        mean = tl.sum(x, axis=0) / H
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / H
        inv_std = tl.rsqrt(var + 1.0e-5)

        gamma = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
        beta = tl.load(bias + offs, mask=mask, other=0.0).to(tl.float32)
        y = centered * inv_std * gamma + beta
        tl.store(out + row * H + offs, y, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@oracle_impl(hardware="H100", shapes="(T([8008, 2560], f32), T([16, 128], i64, gen=Index(8008)), T([128, 2560], f32), T([2560], f32), T([2560], f32), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]))")
def oracle_forward(inputs):
    token_table, token_ids, position_table, weight, bias, *_ = inputs
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    rows = token_ids.numel()
    hidden = weight.numel()
    out = torch.empty((rows, hidden), device=token_table.device, dtype=token_table.dtype)
    block_h = _next_power_of_2(hidden)
    _embedding_layernorm_kernel[(rows,)](
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        out,
        H=hidden,
        POSITIONS=position_table.shape[0],
        BLOCK_H=block_h,
        num_warps=8,
    )
    return (out, out, out)


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
