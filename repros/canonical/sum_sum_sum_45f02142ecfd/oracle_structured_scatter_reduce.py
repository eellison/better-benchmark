"""
Oracle for repro sum_sum_sum_45f02142ecfd (structured_pool_upsample_backward_reduce, PyTorch-UNet).

Gap diagnosis (classification: SCATTER_REDUCE): this repro is a structured
bilinear upsample-backward scatter feeding BN/ReLU masking and sibling channel
reductions. Inductor currently materializes the scattered [8, 64, 320, 479]
intermediate and schedules the reductions as generic consumers; the missing
optimization is SCATTER_REDUCE support that recognizes the structured
index_put(accumulate=True) pattern and accumulates the channel reductions while
producing or bypassing the scatter result as required.

Pattern:
    Bilinear upsample backward (4x index_put with accumulate=True) -> BN-affine/ReLU mask
    -> 3 channel reductions (sum, sum*centered, final BN-backward linear combination).

Key optimization insight:
    The original graph materializes a [8, 64, 320, 479] scattered intermediate from
    4 separate index_put operations on a [8, 64, 640, 958] source tensor (total ~1.2 GB
    of writes). This intermediate is then gated by BN-affine/ReLU and reduced per channel.

    Since the final operation reduces over batch and spatial dims to produce per-channel
    vectors, we can avoid the scatter entirely. Instead, for each output position
    (n, c, h, w) in the [8, 64, 320, 479] output:
      1. Determine which source positions map to it (the inverse of index_put)
      2. Accumulate their weighted contributions
      3. Apply the BN/ReLU gate check
      4. Accumulate into the per-channel reduction

    However, the simpler and more practical approach is: just compute the index_put
    accumulations directly (they ARE the correct approach for bilinear backward), then
    fuse the BN-backward reductions into a single pass. The big win here is:
    - Compute the 4 index_puts efficiently (they use structured indices from bilinear)
    - Fuse all 3 reductions (sum, sum*centered, final linear combo) into one pass
      over the [8, 64, 320, 479] scattered result rather than 3 separate passes.

    Even bigger: since the final outputs are just per-channel sums, we can accumulate
    the BN backward directly as we scatter, avoiding materializing the full intermediate
    at all. For each source element, compute its destination, check the ReLU mask there,
    and accumulate weighted values into per-channel accumulators.

Implementation:
    torch_fused_bilinear_scatter_reduce: Pure PyTorch implementation that:
    1. Materializes the scatter (unavoidable for correctness with overlapping indices)
    2. Fuses the BN-backward reductions into minimal passes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

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


REPRO_ID = "sum_sum_sum_45f02142ecfd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
# Scale factor from the repro: 1 / (N * H_out * W_out) = 1 / (8 * 320 * 479)
CHANNEL_REDUCTION_SCALE = 8.155010438413362e-07
SHAPE_LABEL = "torchbench_pytorch_unet_train_001_8287cdf2"


def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_sum_sum_45f02142ecfd_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def structured_bilinear_index_put(
    getitem_6: torch.Tensor,
    row_weight: torch.Tensor,
    col_weight: torch.Tensor,
    row_hi: torch.Tensor,
    col_hi: torch.Tensor,
    col_lo: torch.Tensor,
    row_lo: torch.Tensor,
) -> torch.Tensor:
    """Compute the 4 bilinear index_put accumulations matching the repro exactly.

    The repro does:
      src = getitem_6[:, 64:128, :, :-1]  -> [8, 64, 640, 958]
      constant_pad_nd removes the last column -> [8, 64, 640, 958]
      Then computes bilinear weights and their complements, and scatters 4 ways:
        val * row_weight * col_weight       -> index_put at (row_hi, col_hi)
        val * row_weight * (1 - col_weight) -> index_put at (row_hi, col_lo)
        val * (1 - row_weight) * col_weight -> index_put at (row_lo, col_hi)
        val * (1 - row_weight) * (1 - col_weight) -> index_put at (row_lo, col_lo)
    """
    # Source: slice channels 64:128 then pad (remove last col)
    src = getitem_6[:, 64:128, :, :-1]  # [8, 64, 640, 958]

    # Compute the 4 weighted contributions
    # The repro computes these as:
    #   mul_tensor = src * row_weight           (val * rw)
    #   add_tensor = src - mul_tensor = src * (1 - rw)   (val * (1-rw))
    #   mul_tensor_1 = mul_tensor * col_weight  (val * rw * cw)
    #   add_tensor_1 = mul_tensor - mul_tensor_1 = mul_tensor * (1 - cw)  (val * rw * (1-cw))
    #   mul_tensor_2 = add_tensor * col_weight  (val * (1-rw) * cw)
    #   add_tensor_2 = add_tensor - mul_tensor_2 = add_tensor * (1-cw)  (val * (1-rw) * (1-cw))
    val_rw = src * row_weight
    val_1mrw = src - val_rw

    val_rw_cw = val_rw * col_weight
    val_rw_1mcw = val_rw - val_rw_cw
    val_1mrw_cw = val_1mrw * col_weight
    val_1mrw_1mcw = val_1mrw - val_1mrw_cw

    # Output shape
    N, C = src.shape[0], src.shape[1]
    out_H, out_W = 320, 479
    out = torch.zeros((N, C, out_H, out_W), device=src.device, dtype=src.dtype)

    # Scatter all 4 contributions (accumulate=True for overlapping indices)
    # index_put at (row_hi, col_hi) with val_rw_cw
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_hi], val_rw_cw, True)
    # index_put at (row_hi, col_lo) with val_rw_1mcw
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_lo], val_rw_1mcw, True)
    # index_put at (row_lo, col_hi) with val_1mrw_cw
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_hi], val_1mrw_cw, True)
    # index_put at (row_lo, col_lo) with val_1mrw_1mcw
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_lo], val_1mrw_1mcw, True)

    return out


def torch_fused_bilinear_scatter_reduce(
    getitem_6: torch.Tensor,     # f32[8, 128, 640, 959]
    arg128_1: torch.Tensor,      # f32[640, 1] - row weights
    arg127_1: torch.Tensor,      # f32[958] - col weights
    arg124_1: torch.Tensor,      # i64[640, 1] - row_hi indices
    arg126_1: torch.Tensor,      # i64[958] - col_hi indices
    arg125_1: torch.Tensor,      # i64[958] - col_lo indices
    arg123_1: torch.Tensor,      # i64[640, 1] - row_lo indices
    arg120_1: torch.Tensor,      # f32[8, 64, 320, 479] - BN input (x)
    arg121_1: torch.Tensor,      # f32[1, 64, 1, 1] - BN running mean
    arg122_1: torch.Tensor,      # f32[1, 64, 1, 1] - BN inv_std
    arg39_1: torch.Tensor,       # f32[64] - BN weight (gamma)
    arg40_1: torch.Tensor,       # f32[64] - BN bias (beta)
    full: torch.Tensor,          # f32[] - scalar for where masking
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bilinear scatter + BN backward reduction oracle.

    Computes:
      1. Bilinear upsample backward via 4 index_put accumulations -> scatter_result [8,64,320,479]
      2. BN-affine/ReLU gate: where(relu(BN(x)) <= 0, full, scatter_result) -> masked
      3. Three reductions fused into two outputs:
         - out0 = sum(masked * centered, [0,2,3]) * inv_std  (per channel)
         - out1 = sum(normalized_grad * affine_scale, [0,2,3])  (per channel)
    """
    # Phase 1: Compute scatter result (bilinear upsample backward)
    scatter_result = structured_bilinear_index_put(
        getitem_6, arg128_1, arg127_1, arg124_1, arg126_1, arg125_1, arg123_1
    )

    # Phase 2: Compute BN-affine/ReLU mask
    # bn_affine = (arg120_1 - mean) * inv_std * weight + bias
    # The gate checks if relu(bn_affine) <= 0
    sub_tensor = arg120_1 - arg121_1  # centered input for gate computation
    mul_tensor_3 = sub_tensor * arg122_1  # normalized
    weight_4d = arg39_1[None, :, None, None]
    bias_4d = arg40_1[None, :, None, None]
    bn_affine = mul_tensor_3 * weight_4d + bias_4d
    relu_gate = torch.relu(bn_affine)
    le_mask = relu_gate <= 0  # True where gated to full_val

    # Apply mask: where gate is active, use full scalar; else use scatter_result
    full_val = full
    masked = torch.where(le_mask, full_val, scatter_result)

    # Phase 3: BN backward reductions (fused into single pass over masked + centered)
    # The repro computes:
    #   mean_4d = squeeze(arg121_1) -> unsqueeze chain -> [1,64,1,1]
    #   centered = arg120_1 - mean_4d  (same as sub_tensor above conceptually)
    #   sum1 = sum(masked, [0,2,3])  -> [64]
    #   sum2 = sum(masked * centered, [0,2,3])  -> [64]
    #   inv_std_vec = squeeze(arg122_1)  -> [64]
    #
    #   mean_term = sum1 * SCALE -> [64], broadcast to [1,64,1,1]
    #   var_term = sum2 * SCALE * inv_std^2 -> [64], broadcast to [1,64,1,1]
    #   normalized_grad = masked - centered * var_term - mean_term
    #   affine_scale = inv_std_vec * arg39_1  -> [64]
    #   out0 = sum2 * inv_std_vec  (mul_tensor_13)
    #   out1 = sum(normalized_grad * affine_scale, [0,2,3])  (sum_dim_int_list_2)

    # centered for BN backward (reusing sub_tensor = arg120_1 - arg121_1)
    centered = sub_tensor  # [8, 64, 320, 479]
    inv_std_vec = arg122_1.squeeze((0, 2, 3))  # [64]

    # Reduction 1: sum(masked, [0,2,3])
    masked_sum = masked.sum(dim=(0, 2, 3))  # [64]

    # Reduction 2: sum(masked * centered, [0,2,3])
    masked_centered_sum = (masked * centered).sum(dim=(0, 2, 3))  # [64]

    # Output 0: mul_tensor_13 = sum_dim_int_list_1 * squeeze_dims_1
    # = masked_centered_sum * inv_std_vec
    out0 = masked_centered_sum * inv_std_vec

    # Output 1: sum(normalized_grad * affine_scale, [0,2,3])
    # where normalized_grad = masked - centered * var_term_4d - mean_term_4d
    # and affine_scale = inv_std_vec * arg39_1
    mean_term = masked_sum * CHANNEL_REDUCTION_SCALE  # [64]
    var_term = masked_centered_sum * CHANNEL_REDUCTION_SCALE * inv_std_vec * inv_std_vec  # [64]
    affine_scale = inv_std_vec * arg39_1  # [64]

    # normalized_grad = masked - centered * var_term[None,:,None,None] - mean_term[None,:,None,None]
    # out1 = sum(normalized_grad * affine_scale[None,:,None,None], [0,2,3])
    # Expand: sum((masked - centered * var_term - mean_term) * affine_scale, [0,2,3])
    # = affine_scale * (sum(masked) - var_term * sum(centered) - mean_term * N*H*W)
    # But sum(centered, [0,2,3]) is NOT zero in general (it's sum(arg120_1 - mean, [0,2,3]))
    # So let's compute it directly:
    centered_sum = centered.sum(dim=(0, 2, 3))  # [64]
    N_HW = 8 * 320 * 479  # total spatial elements per channel

    # sum(normalized_grad * affine_scale, [0,2,3])
    # = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)
    out1 = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)

    return (out0, out1)


@oracle_impl(hardware="H100", shapes="(T([8, 128, 640, 959], f32), T([640, 1], f32), T([958], f32), T([640, 1], i64, gen=Index(8)), T([958], i64, gen=Index(8)), T([958], i64, gen=Index(8)), T([640, 1], i64, gen=Index(8)), T([8, 64, 320, 479], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return torch_fused_bilinear_scatter_reduce(*inputs)


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
