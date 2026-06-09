"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Reformer LSH routing block by deriving signed argmax buckets from bmm_12 without materializing cat([x, -x]), sorting bucket*4096+position, and carrying both sorted gathers through the RMS mean normalization and cyclic concatenation outputs returned by Repro.forward, whereas Inductor currently materializes the 128-wide signed cat, emits generic argmax and key arithmetic, crosses an external sort boundary, then schedules the gather, mean, rsqrt, permute, and cat epilogues as separate kernels; Inductor cannot do this today because the scheduler/codegen pattern library has no Reformer LSH routing pattern that canonicalizes signed-argmax bucket construction, bounded-key sort, sorted gathers, and the downstream RMS/cyclic-concat consumers across the sort boundary; the fix is NEW_PATTERN: add an Inductor lowering for this Reformer LSH routing idiom that computes signed buckets directly and fuses sorted-gather epilogues for all returned tensors."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import triton

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



REPRO_ID = "argmax_mean_9cf9e9271ff1"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
HEADS = 12
SEQ = 4096
DIM = 64
CHUNKS = 64
SIGNED_BUCKETS = 128
FLAT_CHUNKS = BATCH * HEADS * CHUNKS


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this oracle")


def make_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def prepare_oracle_inputs(*inputs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bmm_12, permute_62, mm_19, *_shape_params = inputs
    return bmm_12, permute_62, mm_19


def _signed_lsh_bucket(bmm_12: torch.Tensor) -> torch.Tensor:
    x = (
        bmm_12.view(HEADS, SEQ, 1, 1, 1, DIM)
        .permute(3, 0, 4, 1, 5, 2)
        .reshape(BATCH, HEADS, SEQ, DIM)
    )
    pos_val, pos_idx = x.max(dim=-1)
    neg_val, neg_idx = (-x).max(dim=-1)
    return torch.where(pos_val >= neg_val, pos_idx, neg_idx + DIM).to(torch.int64)


def oracle_reformer_lsh_routing(
    bmm_12: torch.Tensor,
    permute_62: torch.Tensor,
    mm_19: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bucket = _signed_lsh_bucket(bmm_12)
    positions = torch.arange(SEQ, device=bmm_12.device, dtype=torch.int64).view(BATCH, 1, SEQ)
    sort_keys = bucket * SEQ + positions
    sorted_keys, sorted_pos = torch.sort(sort_keys, dim=-1)
    gather_index = sorted_pos.unsqueeze(-1).expand(BATCH, HEADS, SEQ, DIM)

    gathered = torch.gather(permute_62, 2, gather_index)
    gathered_view = gathered.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    gathered_out = gathered_view.view(FLAT_CHUNKS, CHUNKS, DIM)

    squared = torch.pow(gathered_view, 2)
    mean_square = squared.mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_square + 1e-6)
    normalized = gathered_view * rms
    div_tensor = normalized / torch.full((), 8.0, dtype=torch.float64)
    shifted = torch.cat([div_tensor[:, :, -1:, :, :], div_tensor[:, :, :-1, :, :]], dim=2)
    normalized_out = (
        torch.cat([shifted, div_tensor], dim=3)
        .permute(0, 1, 2, 4, 3)
        .view(FLAT_CHUNKS, DIM, SIGNED_BUCKETS)
    )

    mm_view = mm_19.view(BATCH, SEQ, HEADS, DIM).permute(0, 2, 1, 3)
    gathered_mm = torch.gather(mm_view, 2, gather_index)
    gathered_mm_view = gathered_mm.view(BATCH, HEADS, CHUNKS, CHUNKS, DIM)
    shifted_mm = torch.cat([gathered_mm_view[:, :, -1:, :, :], gathered_mm_view[:, :, :-1, :, :]], dim=2)
    mm_out = torch.cat([shifted_mm, gathered_mm_view], dim=3).view(FLAT_CHUNKS, SIGNED_BUCKETS, DIM)

    return sorted_keys, gathered_out, normalized_out, mm_out


class ReformerLSHRoutingOracle(torch.nn.Module):
    def forward(
        self,
        bmm_12: torch.Tensor,
        permute_62: torch.Tensor,
        mm_19: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return oracle_reformer_lsh_routing(bmm_12, permute_62, mm_19)


def reference_outputs(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _metadata_ok(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.device.type == expected.device.type
        and actual.stride() == expected.stride()
    )


def _compile_model(model: torch.nn.Module, inputs: tuple[Any, ...]) -> torch.nn.Module:
    import torch._dynamo

    torch._dynamo.reset()
    compiled = torch.compile(model)
    with torch.no_grad():
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


@oracle_impl(hardware="H100", shapes="(T([12, 4096, 64], f16), T([1, 12, 4096, 64], f16, stride=(3145728, 64, 768, 1)), T([4096, 768], f16), S([12, 4096, 1, 1, 1, 64]), S([1, 12, 1, 4096, 64]), S([1, 12, 1, 1]), S([1, 12, 4096]), S([1, 12, 4096]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 64, 64]), S([768, 64, 64]), S([1, 12, 64, 64, 128]), S([768, 64, 128]), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([-1, -1, -1, 64]), S([1, 12, -1, 64, 64]), S([1, 12, 64, 128, 64]), S([768, 128, 64]))")
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
