"""
Oracle for repro sum_3ee4028cab37 (structured_pool_upsample_backward_reduce, VGG16).

Gap diagnosis (classification: SCATTER_REDUCE): this repro is a structured
max-pool-backward scatter_add feeding a boolean mask and one channel reduction.
Inductor currently materializes the [16384, 12544] scatter_add output before
applying where and sum as generic consumers; the missing optimization is
SCATTER_REDUCE support that recognizes the pool-backward scatter indices and
accumulates the masked per-channel sum from the source values without building
the full scatter matrix.

Pattern:
    max_pool backward (scatter_add) -> where mask -> sum reduction over [batch, H, W].

    Original computation:
      1. full_default = zeros [16384, 12544]     # (batch*channels) x (out_H * out_W)
      2. scatter_add (pool backward) -> [16384, 12544]
      3. reshape to [128, 128, 112, 112]
      4. where(mask, 0, scatter_result) -> zero where mask is True
      5. sum over [0, 2, 3] -> [128]  (per-channel reduction)

Key optimization insight:
    The scatter_add materializes a [16384, 12544] (~200M element, ~800MB) intermediate
    that is immediately reduced via where+sum. Since the final reduction sums over
    batch and spatial dims, we can avoid materializing the scatter output entirely:

    For each source value at (batch, channel, src_h, src_w):
      - Compute the destination spatial index (position in the 112x112 output grid)
      - Check the boolean mask at (batch, channel, dst_h, dst_w)
      - If mask is False (not masked out), accumulate the value into channel sum

    This turns the scatter + reduce into a gather-mask-reduce over the source tensor,
    reducing memory traffic from ~800MB write + ~800MB read to just reading the source
    (~100MB) + mask lookups.

    Additionally, since where(mask, 0, val) zeros out masked positions before summing,
    any source value whose destination is masked contributes nothing. We simply skip it.

Implementation:
    torch_fused_scatter_reduce: Pure PyTorch using advanced indexing (no Triton).
    Avoids the large intermediate via direct gather from mask.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_3ee4028cab37"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"


def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_3ee4028cab37_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_fused_scatter_reduce(
    getitem_24: "f32[128, 128, 56, 56]",
    arg22_1: "i8[128, 128, 56, 56]",
    arg45_1: "b8[128, 128, 112, 112]",
    full: "f32[]",
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> torch.Tensor:
    """Fused scatter-reduce oracle that avoids materializing the scatter output.

    Strategy:
        Instead of scatter_add -> where -> sum, we:
        1. Compute max pool indices (same as original)
        2. For each source value, compute its destination spatial position in 112x112
        3. Check if the mask is False at that position (meaning value contributes)
        4. Accumulate unmasked values per channel

    Since where(mask, 0, scatter_result) zeros masked positions, and the final sum
    reduces batch and spatial dims, we only need to sum source values whose
    destination positions have mask=False.

    The full scalar is 0 (from the repro: full.default([16384, 12544], 0)), and
    where(mask, full=0, val) means masked positions contribute 0. So the sum is
    simply: sum of source values at unmasked destination positions, per channel.
    """
    N, C, src_H, src_W = getitem_24.shape  # [128, 128, 56, 56]
    out_H, out_W = 112, 112
    full_val = full.item()

    # Compute max pool indices from low-memory offsets
    # kernel_size=[2,2], output_size=[112,112], stride=[2,2], padding=[0,0], dilation=[1,1]
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg22_1, [2, 2], [112, 112], [2, 2], [0, 0], [1, 1]
    )  # [128, 128, 56, 56] with values in [0, 12544) = [0, 112*112)

    # Each index value is a flat position in the 112x112 output grid.
    # Decode to (out_h, out_w) for mask lookup
    idx_h = indices_abs // out_W  # [128, 128, 56, 56]
    idx_w = indices_abs % out_W   # [128, 128, 56, 56]

    # The where operation: where(mask, full_val, scatter_result)
    # - Where mask is True: position contributes full_val (=0) to sum
    # - Where mask is False: position contributes scatter_result value
    #
    # But wait - the scatter_add accumulates source values at destination positions.
    # Multiple source positions may map to the same destination. The final sum
    # reduces over all spatial positions, so duplicate contributions at the same
    # destination all get summed together anyway. The only thing that matters is
    # whether a source value's destination is masked or not.
    #
    # However, we also need to account for full_val contributions at positions that
    # have mask=True. Since full_val=0, those contribute nothing.
    #
    # Actually, if full_val != 0 we need:
    #   Part A: full_val * count(mask==True per channel, across batch and spatial)
    #   Part B: sum of source values whose destination has mask==False
    #
    # For this repro, full_val comes from aten.full.default with value 0, so Part A = 0.
    # But let's be general:

    if full_val == 0.0:
        # Only Part B matters: sum source values at unmasked destinations
        # Look up mask at each source value's destination position
        # mask shape: [128, 128, 112, 112], need to gather at (batch, channel, idx_h, idx_w)

        # Use advanced indexing to look up mask values
        batch_idx = torch.arange(N, device=getitem_24.device).reshape(N, 1, 1, 1).expand_as(idx_h)
        chan_idx = torch.arange(C, device=getitem_24.device).reshape(1, C, 1, 1).expand_as(idx_h)

        # Gather mask at destination positions
        mask_at_dst = arg45_1[batch_idx, chan_idx, idx_h, idx_w]  # [128, 128, 56, 56] bool

        # Where mask is True at destination, source value doesn't contribute
        # Where mask is False at destination, source value contributes
        masked_src = torch.where(mask_at_dst, torch.zeros_like(getitem_24), getitem_24)

        # Sum over batch(0) and spatial(2,3) -> [128] per channel
        result = masked_src.sum(dim=(0, 2, 3))  # [128]
    else:
        # General case with non-zero full_val
        # Part A: full_val * number of True mask positions per channel
        mask_true_count = arg45_1.sum(dim=(0, 2, 3)).float()  # [128]
        part_a = full_val * mask_true_count

        # Part B: sum of source values at unmasked destinations
        batch_idx = torch.arange(N, device=getitem_24.device).reshape(N, 1, 1, 1).expand_as(idx_h)
        chan_idx = torch.arange(C, device=getitem_24.device).reshape(1, C, 1, 1).expand_as(idx_h)
        mask_at_dst = arg45_1[batch_idx, chan_idx, idx_h, idx_w]  # [128, 128, 56, 56] bool

        masked_src = torch.where(mask_at_dst, torch.zeros_like(getitem_24), getitem_24)
        part_b = masked_src.sum(dim=(0, 2, 3))  # [128]

        result = part_a + part_b

    return result


def torch_naive_reference(
    getitem_24, arg22_1, arg45_1, full,
    _shape_param_0, _shape_param_1, _shape_param_2,
) -> torch.Tensor:
    """Naive implementation matching repro.py exactly (for correctness comparison)."""
    full_default = torch.ops.aten.full.default(
        [16384, 12544], 0, dtype=torch.float32, layout=torch.strided,
        device=getitem_24.device, pin_memory=False,
    )
    view_default = torch.ops.aten.view.default(getitem_24, _shape_param_0)
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg22_1, [2, 2], [112, 112], [2, 2], [0, 0], [1, 1]
    )
    view_default_1 = torch.ops.aten.view.default(indices_abs, _shape_param_1)
    scatter_add_default = torch.ops.aten.scatter_add.default(
        full_default, 1, view_default_1, view_default,
    )
    view_default_2 = torch.ops.aten.view.default(scatter_add_default, _shape_param_2)
    where_self = torch.ops.aten.where.self(arg45_1, full, view_default_2)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
    return sum_dim_int_list


def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return torch_fused_scatter_reduce(*inputs)


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
