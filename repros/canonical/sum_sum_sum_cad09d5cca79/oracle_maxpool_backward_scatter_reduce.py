"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete PyTorch-UNet max-pool-backward scatter, skip-add, BN-affine/ReLU gate, and sibling BN-backward channel reductions as a direct Triton gather-mask-reduce over the low-memory 2x2 max-pool offsets without materializing the dense f32 `[2048, 38240]` scatter intermediate, whereas Inductor currently expands the low-memory offsets, materializes `scatter_add`, schedules the ReLU-gated `where`, and then performs the dependent reductions as generic tensor work; Inductor cannot do this today because scheduler/codegen does not recognize max-pool-backward offsets as a structured scatter-reduce producer that can feed the surrounding pointwise gate and BN reduction epilogue directly; the fix is SCATTER_REDUCE: add a structured max-pool-backward scatter-reduce lowering that maps each output cell to its owning pooled-gradient source, applies the skip-add and ReLU mask in the producer tile, accumulates the BN channel summaries, and emits the returned reduction tuple from that fused template."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


N = 8
C = 256
SRC_C = 512
H = 160
W = 239
HW = H * W
PH = 80
PW = 119
PHW = PH * PW
NHW = N * HW
REDUCTION_SCALE = 3.268828451882845e-06
BLOCK_M = 1024
BLOCK_C = 8
BLOCK_TILES = 512
N_TILES = (NHW + BLOCK_M - 1) // BLOCK_M


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_torch(
    getitem_18: torch.Tensor,
    getitem_36: torch.Tensor,
    arg69_1: torch.Tensor,
    arg65_1: torch.Tensor,
    arg66_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg15_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch fallback that matches the repro exactly."""
    slice_tensor = getitem_18[:, :C, :, :]
    scatter_base = torch.zeros((N * C, HW), device=getitem_18.device, dtype=torch.float32)
    flat_grad = getitem_36.view(_shape_param_0)
    flat_indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg69_1, [2, 2], [H, W], [2, 2], [0, 0], [1, 1]
    ).view(_shape_param_1)
    scatter = scatter_base.scatter_add(1, flat_indices, flat_grad).view(_shape_param_2)
    add_tensor = slice_tensor + scatter

    centered = arg65_1 - arg66_1
    affine = centered * arg67_1 * arg14_1[None, :, None, None] + arg15_1[None, :, None, None]
    masked = torch.where(torch.relu(affine) <= 0, full, add_tensor)

    sum0 = masked.sum(dim=(0, 2, 3))
    sum1 = (masked * centered).sum(dim=(0, 2, 3))
    inv_std = arg67_1.squeeze((0, 2, 3))
    out0 = sum1 * inv_std

    mean_term = sum0 * REDUCTION_SCALE
    var_term = sum1 * REDUCTION_SCALE * inv_std * inv_std
    input_scale = inv_std * arg14_1
    out1 = (
        (masked - centered * var_term[None, :, None, None] - mean_term[None, :, None, None])
        * input_scale[None, :, None, None]
    ).sum(dim=(0, 2, 3))
    return out0, out1


if triton is not None:

    @triton.jit
    def _gather_mask_partial_kernel(
        getitem18_ptr,
        getitem36_ptr,
        offsets_ptr,
        arg65_ptr,
        arg66_ptr,
        arg67_ptr,
        arg14_ptr,
        arg15_ptr,
        full_ptr,
        partial_ptr,
        C_: tl.constexpr,
        SRC_C_: tl.constexpr,
        HW_: tl.constexpr,
        W_: tl.constexpr,
        PW_: tl.constexpr,
        PHW_: tl.constexpr,
        NHW_: tl.constexpr,
        N_TILES_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        tile_id = tl.program_id(1)

        offs_m = tile_id * BLOCK_M_ + tl.arange(0, BLOCK_M_)[:, None]
        offs_c_vec = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        offs_c = offs_c_vec[None, :]
        mask = (offs_m < NHW_) & (offs_c < C_)

        n_idx = offs_m // HW_
        hw_idx = offs_m - n_idx * HW_
        h_idx = hw_idx // W_
        w_idx = hw_idx - h_idx * W_

        getitem18_offsets = n_idx * (SRC_C_ * HW_) + offs_c * HW_ + hw_idx
        arg65_offsets = n_idx * (C_ * HW_) + offs_c * HW_ + hw_idx

        slice_vals = tl.load(getitem18_ptr + getitem18_offsets, mask=mask, other=0.0).to(tl.float32)
        arg65_vals = tl.load(arg65_ptr + arg65_offsets, mask=mask, other=0.0).to(tl.float32)

        pool_h = h_idx // 2
        pool_w = w_idx // 2
        local_offset = (h_idx - pool_h * 2) * 2 + (w_idx - pool_w * 2)
        pool_offsets = n_idx * (C_ * PHW_) + offs_c * PHW_ + pool_h * PW_ + pool_w
        pool_domain = mask & (w_idx < (PW_ * 2))
        pool_offset_vals = tl.load(offsets_ptr + pool_offsets, mask=pool_domain, other=-1).to(tl.int32)
        selected = pool_domain & (pool_offset_vals == local_offset)
        pool_vals = tl.load(getitem36_ptr + pool_offsets, mask=selected, other=0.0).to(tl.float32)

        channel_mask = offs_c_vec < C_
        mean = tl.load(arg66_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)
        inv_std = tl.load(arg67_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)
        weight = tl.load(arg14_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)
        bias = tl.load(arg15_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)
        full_val = tl.load(full_ptr).to(tl.float32)

        centered = arg65_vals - mean[None, :]
        affine = centered * inv_std[None, :] * weight[None, :] + bias[None, :]
        scatter_add = slice_vals + pool_vals
        masked = tl.where(affine <= 0.0, full_val, scatter_add)
        masked = tl.where(mask, masked, 0.0)
        centered = tl.where(mask, centered, 0.0)

        sum0 = tl.sum(masked, axis=0)
        sum1 = tl.sum(masked * centered, axis=0)
        sum_centered = tl.sum(centered, axis=0)

        out_offsets = tile_id * C_ + offs_c_vec
        tl.store(partial_ptr + out_offsets, sum0, mask=channel_mask)
        tl.store(partial_ptr + N_TILES_ * C_ + out_offsets, sum1, mask=channel_mask)
        tl.store(partial_ptr + 2 * N_TILES_ * C_ + out_offsets, sum_centered, mask=channel_mask)

    @triton.jit
    def _finalize_kernel(
        partial_ptr,
        arg67_ptr,
        arg14_ptr,
        out0_ptr,
        out1_ptr,
        C_: tl.constexpr,
        N_TILES_: tl.constexpr,
        NHW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        offs_t = tl.arange(0, BLOCK_TILES_)[:, None]
        offs_c_vec = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        offs_c = offs_c_vec[None, :]
        mask = (offs_t < N_TILES_) & (offs_c < C_)
        offsets = offs_t * C_ + offs_c

        sum0 = tl.sum(tl.load(partial_ptr + offsets, mask=mask, other=0.0), axis=0)
        sum1 = tl.sum(tl.load(partial_ptr + N_TILES_ * C_ + offsets, mask=mask, other=0.0), axis=0)
        sum_centered = tl.sum(
            tl.load(partial_ptr + 2 * N_TILES_ * C_ + offsets, mask=mask, other=0.0),
            axis=0,
        )

        channel_mask = offs_c_vec < C_
        inv_std = tl.load(arg67_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)
        weight = tl.load(arg14_ptr + offs_c_vec, mask=channel_mask, other=0.0).to(tl.float32)

        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * inv_std * inv_std
        input_scale = inv_std * weight
        out0 = sum1 * inv_std
        out1 = input_scale * (sum0 - var_term * sum_centered - mean_term * NHW_)

        tl.store(out0_ptr + offs_c_vec, out0, mask=channel_mask)
        tl.store(out1_ptr + offs_c_vec, out1, mask=channel_mask)


def oracle_triton(
    getitem_18: torch.Tensor,
    getitem_36: torch.Tensor,
    arg69_1: torch.Tensor,
    arg65_1: torch.Tensor,
    arg66_1: torch.Tensor,
    arg67_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg15_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_18.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if tuple(getitem_18.shape) != (N, SRC_C, H, W) or tuple(getitem_36.shape) != (N, C, PH, PW):
        raise ValueError("unexpected repro shape")

    partial = torch.empty((3, N_TILES, C), device=getitem_18.device, dtype=torch.float32)
    out0 = torch.empty((C,), device=getitem_18.device, dtype=torch.float32)
    out1 = torch.empty((C,), device=getitem_18.device, dtype=torch.float32)

    _gather_mask_partial_kernel[(triton.cdiv(C, BLOCK_C), N_TILES)](
        getitem_18,
        getitem_36,
        arg69_1,
        arg65_1,
        arg66_1,
        arg67_1,
        arg14_1,
        arg15_1,
        full,
        partial,
        C_=C,
        SRC_C_=SRC_C,
        HW_=HW,
        W_=W,
        PW_=PW,
        PHW_=PHW,
        NHW_=NHW,
        N_TILES_=N_TILES,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=8,
    )
    _finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial,
        arg67_1,
        arg14_1,
        out0,
        out1,
        C_=C,
        N_TILES_=N_TILES,
        NHW_=NHW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_TILES_=BLOCK_TILES,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    return out0, out1


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    getitem_18 = inputs[0]
    if isinstance(getitem_18, torch.Tensor) and getitem_18.device.type == "cuda" and triton is not None:
        return oracle_triton(*inputs)
    return oracle_torch(*inputs)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
