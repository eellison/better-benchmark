"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete LayerNorm-backward return tuple by row-tiling the `[1152000, 512]` producer, preserving the fp32 `dy * gamma` row reduction inside the producer and using Triton to finalize fp32 `[512]` column partials, whereas Inductor materializes a row-invariant `[1152000, 1]` `sum(dy * gamma)` buffer before its mix-order reduction; Inductor cannot do this today because its algebraic simplifier does not prove that a reduction over an expanded scalar times a channel vector is independent of the row dimension once it feeds a later multi-output reduction; the fix is ALGEBRAIC_ELIMINATION: canonicalize broadcast-only row reductions to scalar/channel summaries and pass them directly into the fused LayerNorm-backward producer."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

from oracle_harness import oracle_impl


ROWS = 1_152_000
CHANNELS = 512
ROW_TILE = 64
FINAL_BLOCK_CHANNELS = 16
PARTIAL_REDUCE_BLOCK_TILES = 256
PARTIAL_REDUCE_BLOCK_CHANNELS = 16
FINAL_STAGE_BLOCK_TILES = 128


triton_helpers.set_driver_to_gpu()


@triton_heuristics.persistent_reduction(
    size_hints={"x": 2097152, "r0_": 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        "signature": {
            "dy_scalar_ptr": "*bf16",
            "gamma_ptr": "*fp32",
            "x_ptr": "*bf16",
            "mean_ptr": "*fp32",
            "invstd_ptr": "*fp32",
            "out_grad_ptr": "*bf16",
            "partial_sum_ptr": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
            "RSPLIT_SIZE": "constexpr",
            "NUM_STAGES": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda", 0)),
        "constants": {},
        "native_matmul": False,
        "enable_fp_fusion": True,
        "launch_pdl": False,
        "disable_ftz": False,
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
                (4,): [["tt.divisibility", 16]],
                (5,): [["tt.divisibility", 16]],
                (6,): [["tt.divisibility", 16]],
                (7,): [["tt.divisibility", 16]],
                (8,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "MixOrderReductionGrid",
        "kernel_name": "oracle_layernorm_bwd_mix_order",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": None,
        "atomic_add_found": False,
        "num_load": 5,
        "num_store": 0,
        "num_reduction": 1,
        "autotune_hints": set(),
        "RSPLIT_SIZE": 64,
        "backend_hash": "oracle",
        "assert_indirect_indexing": True,
        "autotune_local_cache": False,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "incremental_autotune": False,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
        "deterministic": False,
        "batch_invariant": False,
        "force_filter_reduction_configs": False,
        "mix_order_reduction_allow_multi_stages": True,
        "dynamic_disable_pipelining": True,
        "are_deterministic_algorithms_enabled": False,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def _layernorm_backward_partial_kernel(
    dy_scalar_ptr,
    gamma_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    out_grad_ptr,
    partial_sum_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    RSPLIT_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    xnumel = 1152000
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * RSPLIT_SIZE
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    accum0 = tl.full([R0_BLOCK], 0, tl.float32)[None, :]
    dy_scalar = tl.load(dy_scalar_ptr + 0).to(tl.float32)
    dy = tl.broadcast_to(dy_scalar, [1, 1])
    gamma = tl.load(gamma_ptr + r0_index, None, eviction_policy="evict_last")
    dy_f32 = dy.to(tl.float32)
    dy_gamma = dy_f32 * gamma
    dy_gamma_scaled = dy_gamma * 512.0
    dy_gamma_sum = tl.sum(dy_gamma, 1)[:, None].to(tl.float32)
    split_size = min(RSPLIT_SIZE, xnumel - xoffset)
    for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):
        xmask = xindex < xnumel
        row = xindex
        xindex += XBLOCK
        x = tl.load(
            x_ptr + (r0_index + 512 * row),
            xmask,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(mean_ptr + row, xmask, eviction_policy="evict_last")
        invstd = tl.load(invstd_ptr + row, xmask, eviction_policy="evict_last")
        centered = x.to(tl.float32) - mean
        normed = centered * invstd
        row_products = dy_gamma * normed
        row_products = tl.broadcast_to(row_products, [XBLOCK, R0_BLOCK])
        row_dot = tl.sum(tl.where(xmask, row_products, 0.0), 1)[:, None].to(tl.float32)
        invstd_div = invstd * 0.001953125
        centered_dy = dy_gamma_scaled - dy_gamma_sum
        grad = invstd_div * (centered_dy - normed * row_dot)
        tl.store(out_grad_ptr + (r0_index + 512 * row), grad.to(tl.float32), xmask)
        col_sum = tl.sum(dy_f32 * normed, 0)
        accum0 += col_sum
    tl.store(
        partial_sum_ptr + tl.program_id(0) * r0_numel + r0_index,
        accum0,
        tl.full([R0_BLOCK], True, tl.int1)[None, :],
    )


@triton.jit
def _reduce_partials_stage_kernel(
    partial_sum_ptr,
    stage_sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS_: tl.constexpr,
):
    tile_block = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
    tiles = tile_block * BLOCK_TILES + tl.arange(0, BLOCK_TILES)
    mask = tiles[:, None] < NUM_ROW_TILES
    offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
    values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    reduced = tl.sum(values, axis=0)
    tl.store(stage_sum_ptr + tile_block * CHANNELS_ + cols, reduced)


@triton.jit
def _finalize_stage_sum_kernel(
    stage_sum_ptr,
    out_sum_ptr,
    NUM_STAGE_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS_: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles[:, None] < NUM_STAGE_TILES
    offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
    values = tl.load(stage_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + cols, tl.sum(values, axis=0))


# 4e21884e: (T([1152000,1], f32), T([], bf16), T([512], f32), T([1152000,512], bf16), T([1152000,1], f32), S([1152000,512]))
@oracle_impl(hardware="B200", point="4e21884e")
def oracle_forward(inputs):
    invstd, dy_scalar, gamma, x, mean, shape_param = inputs
    num_row_tiles = triton.cdiv(ROWS, ROW_TILE)
    out_grad = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out_sum = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    partial_sum = torch.empty(
        (num_row_tiles, CHANNELS),
        device=x.device,
        dtype=torch.float32,
    )
    num_stage_tiles = triton.cdiv(num_row_tiles, PARTIAL_REDUCE_BLOCK_TILES)
    stage_sum = torch.empty(
        (num_stage_tiles, CHANNELS),
        device=x.device,
        dtype=torch.float32,
    )

    stream = get_raw_stream(x.device.index if x.device.index is not None else 0)
    _layernorm_backward_partial_kernel.run(
        dy_scalar,
        gamma,
        x,
        mean,
        invstd,
        out_grad,
        partial_sum,
        ROWS,
        CHANNELS,
        stream=stream,
    )
    _reduce_partials_stage_kernel[
        (
            num_stage_tiles,
            triton.cdiv(CHANNELS, PARTIAL_REDUCE_BLOCK_CHANNELS),
        )
    ](
        partial_sum,
        stage_sum,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=PARTIAL_REDUCE_BLOCK_TILES,
        BLOCK_CHANNELS_=PARTIAL_REDUCE_BLOCK_CHANNELS,
        num_warps=8,
    )
    _finalize_stage_sum_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        stage_sum,
        out_sum,
        NUM_STAGE_TILES=num_stage_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_STAGE_BLOCK_TILES,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )
    return out_grad, out_sum
