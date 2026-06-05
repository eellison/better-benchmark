"""
Oracle for argmax_mean_180a59791f51: Reformer LSH routing and post-sort gather.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the Reformer LSH
route key directly as a signed argmax over the original 64-vector, using max(x)
and max(-x) to produce the same 0..127 bucket without first materializing the
128-wide `cat([x, -x])`, then sorts `bucket * 4096 + position` and applies the
same two sorted gathers, RMS scaling, and cyclic concatenations as the captured
graph; Inductor currently emits a pointwise kernel that writes the full
cat/negation buffer, a separate argmax/key kernel, an external sort, and
post-sort kernels because the scheduler does not canonicalize
`argmax(cat([x, -x]))` plus bounded-key sort arithmetic into an LSH routing
operation and the external sort boundary prevents this algebra from being
pulled into the consumers. The Inductor fix is a NEW_PATTERN lowering for this
Reformer LSH routing idiom: signed-abs argmax key generation, bounded-key
position sort, and fused sorted gather/RMS/cyclic-concat epilogue without the
materialized cat input.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "argmax_mean_180a59791f51"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
SEQ = 4096
DIM = 64
CHUNKS = 64
SIGNED_BUCKETS = 128
FLAT_CHUNKS = BATCH * HEADS * CHUNKS

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


def _signed_lsh_bucket(bmm: torch.Tensor) -> torch.Tensor:
    x = (
        bmm.view(HEADS, BATCH, SEQ, 1, 1, DIM)
        .permute(1, 0, 3, 2, 5, 4)
        .reshape(BATCH, HEADS, SEQ, DIM)
    )
    pos_val, pos_idx = x.max(dim=-1)
    neg_val, neg_idx = (-x).max(dim=-1)
    return torch.where(pos_val >= neg_val, pos_idx, neg_idx + DIM).to(torch.int64)


def oracle_reformer_lsh_routing(
    bmm: torch.Tensor,
    permute_1: torch.Tensor,
    mm_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bucket = _signed_lsh_bucket(bmm)
    positions = torch.arange(SEQ, device=bmm.device, dtype=torch.int64).view(1, 1, SEQ)
    keys = bucket * SEQ + positions
    sorted_keys, sorted_pos = torch.sort(keys, dim=-1)
    gather_index = sorted_pos.unsqueeze(-1).expand(BATCH, HEADS, SEQ, DIM)

    gathered = torch.gather(permute_1, 2, gather_index)
    gathered_view = gathered.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    gathered_out = gathered_view.view(FLAT_CHUNKS, CHUNKS, DIM)

    rms_scale = torch.rsqrt((gathered_view * gathered_view).mean(dim=-1, keepdim=True) + 1e-6) / 8.0
    normalized = gathered_view * rms_scale
    shifted = torch.cat([normalized[:, :, -1:, :, :], normalized[:, :, :-1, :, :]], dim=2)
    normalized_out = (
        torch.cat([shifted, normalized], dim=3)
        .permute(0, 1, 2, 4, 3)
        .reshape(FLAT_CHUNKS, DIM, SIGNED_BUCKETS)
    )

    mm_view = mm_1.view(BATCH, SEQ, HEADS, DIM).permute(0, 2, 1, 3)
    gathered_mm = torch.gather(mm_view, 2, gather_index)
    gathered_mm_view = gathered_mm.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    shifted_mm = torch.cat([gathered_mm_view[:, :, -1:, :, :], gathered_mm_view[:, :, :-1, :, :]], dim=2)
    mm_out = torch.cat([shifted_mm, gathered_mm_view], dim=3).reshape(FLAT_CHUNKS, SIGNED_BUCKETS, DIM)

    return (
        sorted_keys,
        bucket.view(BATCH, HEADS, 1, SEQ),
        gathered_out,
        normalized_out,
        mm_out,
    )


class ReformerLSHRoutingOracle(torch.nn.Module):
    def forward(
        self,
        bmm: torch.Tensor,
        permute_1: torch.Tensor,
        mm_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return oracle_reformer_lsh_routing(bmm, permute_1, mm_1)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bmm, permute_1, mm_1, *_shape_params = inputs
    return bmm, permute_1, mm_1


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


def oracle_forward(inputs):
    return oracle_reformer_lsh_routing(*inputs)


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
