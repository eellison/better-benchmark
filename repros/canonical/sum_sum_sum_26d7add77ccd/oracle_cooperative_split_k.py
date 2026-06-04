"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Adv-Inception avg-pool-add fanout BN-backward return tuple with one channel-persistent Triton producer that reconstructs the structured avg_pool2d_backward plus sibling-add source, performs each branch's ReLU-gated `(N, H, W)` reductions, and writes all six tensor gradients plus six scale-gradient vectors, whereas Inductor currently lowers the avg-pool backward, three full-tensor adds, channel slices, six ReLU masks, twelve `sum([0, 2, 3])` reductions, and dependent BN-backward epilogues as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative multi-output reduction template that shares a structured pooled-add producer across disjoint channel slices while feeding both sibling channel sums and full-tensor side-output epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split or persist the shared `(N, H, W)` reduction domain by channel slice, fuse structured avg-pool-backward/add producers into the reduction, and finalize each branch's tensor and vector outputs from the same coordinated accumulators."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_26d7add77ccd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
MM_C = 2048
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
AVG_SCALE = 1.0 / 9.0
REDUCE_SCALE = 1.0 / TOTAL_SPATIAL
SINGLE_PASS_WARPS = 16



@triton.jit
def _pooled_add_source(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    base,
    h,
    w,
    active,
    H_: tl.constexpr,
    W_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
):
    up = h > 0
    down = h < H_ - 1
    left = w > 0
    right = w < W_ - 1

    pool = tl.load(pool_grad_ptr + base, mask=active, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base - W_ - 1, mask=active & up & left, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base - W_, mask=active & up, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base - W_ + 1, mask=active & up & right, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base - 1, mask=active & left, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base + 1, mask=active & right, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base + W_ - 1, mask=active & down & left, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base + W_, mask=active & down, other=0.0).to(tl.float32)
    pool += tl.load(pool_grad_ptr + base + W_ + 1, mask=active & down & right, other=0.0).to(tl.float32)
    pool *= AVG_SCALE_

    source = pool + tl.load(add0_ptr + base, mask=active, other=0.0).to(tl.float32)
    source += tl.load(add1_ptr + base, mask=active, other=0.0).to(tl.float32)
    source += tl.load(add2_ptr + base, mask=active, other=0.0).to(tl.float32)
    return source


@triton.jit
def _bn_branch_partial_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    activation_input_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    source_offset: tl.constexpr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    MM_C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    branch_offsets = n[:, None] * (C_ * HW_) + c[None, :] * HW_ + hw[:, None]
    source_base = n[:, None] * (MM_C_ * HW_) + (source_offset + c[None, :]) * HW_ + hw[:, None]
    active = k_mask[:, None] & c_mask[None, :]

    x = tl.load(activation_input_ptr + branch_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)
    source = _pooled_add_source(
        pool_grad_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        source_base,
        h[:, None],
        w[:, None],
        active,
        H_=H_,
        W_=W_,
        AVG_SCALE_=AVG_SCALE_,
    )

    centered = x - mean[None, :]
    relu_input = centered * invstd[None, :] * gamma[None, :] + beta[None, :]
    where_self = tl.where(relu_input <= 0.0, full, source)
    where_self = tl.where(active, where_self, 0.0)

    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(where_self * centered, axis=0)
    partial_offsets = c * num_tiles + tl.program_id(1)
    tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)
    tl.store(partial_sum2_ptr + partial_offsets, sum2, mask=c_mask)


@triton.jit
def _bn_branch_finalize_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < C_
    tile_mask = tiles < num_tiles
    offsets = tiles[:, None] + c[None, :] * num_tiles

    values1 = tl.load(partial_sum1_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    values2 = tl.load(partial_sum2_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    sum1 = tl.sum(values1, axis=0)
    sum2 = tl.sum(values2, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum1_ptr + c, sum1, mask=c_mask)
    tl.store(sum2_ptr + c, sum2, mask=c_mask)
    tl.store(vector_out_ptr + c, sum2 * invstd, mask=c_mask)


@triton.jit
def _bn_branch_epilogue_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    activation_input_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    sum1_ptr,
    sum2_ptr,
    tensor_out_ptr,
    source_offset: tl.constexpr,
    numel: tl.constexpr,
    C_: tl.constexpr,
    MM_C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < numel

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    h = hw // W_
    w = hw - h * W_
    source_base = n * (MM_C_ * HW_) + (source_offset + c) * HW_ + hw

    x = tl.load(activation_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=active, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)
    source = _pooled_add_source(
        pool_grad_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        source_base,
        h,
        w,
        active,
        H_=H_,
        W_=W_,
        AVG_SCALE_=AVG_SCALE_,
    )

    centered = x - mean
    relu_input = centered * invstd * gamma + beta
    where_self = tl.where(relu_input <= 0.0, full, source)
    variance_term = sum2 * REDUCE_SCALE_ * invstd * invstd
    mean_term = sum1 * REDUCE_SCALE_
    out = (where_self - centered * variance_term - mean_term) * (invstd * gamma)
    tl.store(tensor_out_ptr + offsets, out, mask=active)


@triton.jit
def _all_branches_single_pass_pooled_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    x_a_ptr,
    mean_a_ptr,
    invstd_a_ptr,
    gamma_a_ptr,
    beta_a_ptr,
    x_b_ptr,
    mean_b_ptr,
    invstd_b_ptr,
    gamma_b_ptr,
    beta_b_ptr,
    x_c_ptr,
    mean_c_ptr,
    invstd_c_ptr,
    gamma_c_ptr,
    beta_c_ptr,
    x_d_ptr,
    mean_d_ptr,
    invstd_d_ptr,
    gamma_d_ptr,
    beta_d_ptr,
    x_e_ptr,
    mean_e_ptr,
    invstd_e_ptr,
    gamma_e_ptr,
    beta_e_ptr,
    x_f_ptr,
    mean_f_ptr,
    invstd_f_ptr,
    gamma_f_ptr,
    beta_f_ptr,
    full_ptr,
    out_a_ptr,
    vec_a_ptr,
    out_b_ptr,
    vec_b_ptr,
    out_c_ptr,
    vec_c_ptr,
    out_d_ptr,
    vec_d_ptr,
    out_e_ptr,
    vec_e_ptr,
    out_f_ptr,
    vec_f_ptr,
    MM_C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    source_base = n * (MM_C_ * HW_) + c * HW_ + hw
    source = _pooled_add_source(
        pool_grad_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        source_base,
        h,
        w,
        k_mask,
        H_=H_,
        W_=W_,
        AVG_SCALE_=AVG_SCALE_,
    )

    is_f = c < 320
    is_e = (c >= 320) & (c < 704)
    is_d = (c >= 704) & (c < 1088)
    is_c = (c >= 1088) & (c < 1472)
    is_b = (c >= 1472) & (c < 1856)
    is_a = c >= 1856
    local_c = tl.where(
        is_f,
        c,
        tl.where(
            is_e,
            c - 320,
            tl.where(
                is_d,
                c - 704,
                tl.where(is_c, c - 1088, tl.where(is_b, c - 1472, c - 1856)),
            ),
        ),
    )

    x = tl.zeros((BLOCK_K,), dtype=tl.float32)
    mean = tl.zeros((), dtype=tl.float32)
    invstd = tl.zeros((), dtype=tl.float32)
    gamma = tl.zeros((), dtype=tl.float32)
    beta = tl.zeros((), dtype=tl.float32)

    offsets_f = n * (320 * HW_) + local_c * HW_ + hw
    x += tl.load(x_f_ptr + offsets_f, mask=k_mask & is_f, other=0.0).to(tl.float32)
    mean += tl.load(mean_f_ptr + local_c, mask=is_f, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_f_ptr + local_c, mask=is_f, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_f_ptr + local_c, mask=is_f, other=0.0).to(tl.float32)
    beta += tl.load(beta_f_ptr + local_c, mask=is_f, other=0.0).to(tl.float32)

    offsets_e = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x_e_ptr + offsets_e, mask=k_mask & is_e, other=0.0).to(tl.float32)
    mean += tl.load(mean_e_ptr + local_c, mask=is_e, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_e_ptr + local_c, mask=is_e, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_e_ptr + local_c, mask=is_e, other=0.0).to(tl.float32)
    beta += tl.load(beta_e_ptr + local_c, mask=is_e, other=0.0).to(tl.float32)

    offsets_d = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x_d_ptr + offsets_d, mask=k_mask & is_d, other=0.0).to(tl.float32)
    mean += tl.load(mean_d_ptr + local_c, mask=is_d, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_d_ptr + local_c, mask=is_d, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_d_ptr + local_c, mask=is_d, other=0.0).to(tl.float32)
    beta += tl.load(beta_d_ptr + local_c, mask=is_d, other=0.0).to(tl.float32)

    offsets_c = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x_c_ptr + offsets_c, mask=k_mask & is_c, other=0.0).to(tl.float32)
    mean += tl.load(mean_c_ptr + local_c, mask=is_c, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_c_ptr + local_c, mask=is_c, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_c_ptr + local_c, mask=is_c, other=0.0).to(tl.float32)
    beta += tl.load(beta_c_ptr + local_c, mask=is_c, other=0.0).to(tl.float32)

    offsets_b = n * (384 * HW_) + local_c * HW_ + hw
    x += tl.load(x_b_ptr + offsets_b, mask=k_mask & is_b, other=0.0).to(tl.float32)
    mean += tl.load(mean_b_ptr + local_c, mask=is_b, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_b_ptr + local_c, mask=is_b, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_b_ptr + local_c, mask=is_b, other=0.0).to(tl.float32)
    beta += tl.load(beta_b_ptr + local_c, mask=is_b, other=0.0).to(tl.float32)

    offsets_a = n * (192 * HW_) + local_c * HW_ + hw
    x += tl.load(x_a_ptr + offsets_a, mask=k_mask & is_a, other=0.0).to(tl.float32)
    mean += tl.load(mean_a_ptr + local_c, mask=is_a, other=0.0).to(tl.float32)
    invstd += tl.load(invstd_a_ptr + local_c, mask=is_a, other=0.0).to(tl.float32)
    gamma += tl.load(gamma_a_ptr + local_c, mask=is_a, other=0.0).to(tl.float32)
    beta += tl.load(beta_a_ptr + local_c, mask=is_a, other=0.0).to(tl.float32)

    centered = x - mean
    relu_input = centered * invstd * gamma + beta
    full = tl.load(full_ptr).to(tl.float32)
    where_self = tl.where(relu_input <= 0.0, full, source)
    where_self = tl.where(k_mask, where_self, 0.0)
    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(where_self * centered, axis=0)
    variance_term = sum2 * REDUCE_SCALE_ * invstd * invstd
    mean_term = sum1 * REDUCE_SCALE_
    out = (where_self - centered * variance_term - mean_term) * (invstd * gamma)
    vector_out = sum2 * invstd

    tl.store(out_f_ptr + offsets_f, out, mask=k_mask & is_f)
    tl.store(vec_f_ptr + local_c, vector_out, mask=is_f)
    tl.store(out_e_ptr + offsets_e, out, mask=k_mask & is_e)
    tl.store(vec_e_ptr + local_c, vector_out, mask=is_e)
    tl.store(out_d_ptr + offsets_d, out, mask=k_mask & is_d)
    tl.store(vec_d_ptr + local_c, vector_out, mask=is_d)
    tl.store(out_c_ptr + offsets_c, out, mask=k_mask & is_c)
    tl.store(vec_c_ptr + local_c, vector_out, mask=is_c)
    tl.store(out_b_ptr + offsets_b, out, mask=k_mask & is_b)
    tl.store(vec_b_ptr + local_c, vector_out, mask=is_b)
    tl.store(out_a_ptr + offsets_a, out, mask=k_mask & is_a)
    tl.store(vec_a_ptr + local_c, vector_out, mask=is_a)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def _run_bn_branch(
    pool_grad: torch.Tensor,
    add0: torch.Tensor,
    add1: torch.Tensor,
    add2: torch.Tensor,
    activation_input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    full: torch.Tensor,
    source_offset: int,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert pool_grad.shape == (N, MM_C, H, W)
    assert add0.shape == (N, MM_C, H, W)
    assert add1.shape == (N, MM_C, H, W)
    assert add2.shape == (N, MM_C, H, W)
    assert activation_input.shape == (N, channels, H, W)
    assert mean.shape == (1, channels, 1, 1)
    assert invstd.shape == (1, channels, 1, 1)
    assert gamma.shape == (channels,)
    assert beta.shape == (channels,)
    assert pool_grad.is_contiguous()
    assert add0.is_contiguous()
    assert add1.is_contiguous()
    assert add2.is_contiguous()
    assert activation_input.is_contiguous()

    device = pool_grad.device
    block_c = 16
    block_k = 512
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=device, dtype=torch.float32)
    sum2 = torch.empty((channels,), device=device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=device, dtype=torch.float32)

    _bn_branch_partial_kernel[(triton.cdiv(channels, block_c), num_tiles)](
        pool_grad,
        add0,
        add1,
        add2,
        activation_input,
        mean,
        invstd,
        gamma,
        beta,
        full,
        partial_sum1,
        partial_sum2,
        source_offset=source_offset,
        num_tiles=num_tiles,
        C_=channels,
        MM_C_=MM_C,
        H_=H,
        W_=W,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        AVG_SCALE_=AVG_SCALE,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=8,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _bn_branch_finalize_kernel[(triton.cdiv(channels, block_c),)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        C_=channels,
        BLOCK_C=block_c,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    tensor_out = torch.empty_like(activation_input)
    numel = N * channels * HW
    block_elems = 256
    _bn_branch_epilogue_kernel[(triton.cdiv(numel, block_elems),)](
        pool_grad,
        add0,
        add1,
        add2,
        activation_input,
        mean,
        invstd,
        gamma,
        beta,
        full,
        sum1,
        sum2,
        tensor_out,
        source_offset=source_offset,
        numel=numel,
        C_=channels,
        MM_C_=MM_C,
        H_=H,
        W_=W,
        HW_=HW,
        AVG_SCALE_=AVG_SCALE,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )
    return tensor_out, vector_out


def oracle_full(
    getitem: torch.Tensor,
    arg514_1: torch.Tensor,
    getitem_12: torch.Tensor,
    getitem_21: torch.Tensor,
    getitem_24: torch.Tensor,
    arg511_1: torch.Tensor,
    arg512_1: torch.Tensor,
    arg513_1: torch.Tensor,
    arg209_1: torch.Tensor,
    arg210_1: torch.Tensor,
    full_1: torch.Tensor,
    arg507_1: torch.Tensor,
    arg508_1: torch.Tensor,
    arg509_1: torch.Tensor,
    arg206_1: torch.Tensor,
    arg207_1: torch.Tensor,
    arg504_1: torch.Tensor,
    arg505_1: torch.Tensor,
    arg506_1: torch.Tensor,
    arg203_1: torch.Tensor,
    arg204_1: torch.Tensor,
    arg495_1: torch.Tensor,
    arg496_1: torch.Tensor,
    arg497_1: torch.Tensor,
    arg196_1: torch.Tensor,
    arg197_1: torch.Tensor,
    arg492_1: torch.Tensor,
    arg493_1: torch.Tensor,
    arg494_1: torch.Tensor,
    arg193_1: torch.Tensor,
    arg194_1: torch.Tensor,
    arg486_1: torch.Tensor,
    arg487_1: torch.Tensor,
    arg488_1: torch.Tensor,
    arg188_1: torch.Tensor,
    arg189_1: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    assert arg514_1.shape == (N, MM_C, H, W)

    pool_grad = getitem.contiguous()
    add0 = getitem_12.contiguous()
    add1 = getitem_21.contiguous()
    add2 = getitem_24.contiguous()
    full = full_1.contiguous()

    out_a_tensor = torch.empty_like(arg511_1)
    out_a_vec = torch.empty((192,), device=pool_grad.device, dtype=torch.float32)
    out_b_tensor = torch.empty_like(arg507_1)
    out_b_vec = torch.empty((384,), device=pool_grad.device, dtype=torch.float32)
    out_c_tensor = torch.empty_like(arg504_1)
    out_c_vec = torch.empty((384,), device=pool_grad.device, dtype=torch.float32)
    out_d_tensor = torch.empty_like(arg495_1)
    out_d_vec = torch.empty((384,), device=pool_grad.device, dtype=torch.float32)
    out_e_tensor = torch.empty_like(arg492_1)
    out_e_vec = torch.empty((384,), device=pool_grad.device, dtype=torch.float32)
    out_f_tensor = torch.empty_like(arg486_1)
    out_f_vec = torch.empty((320,), device=pool_grad.device, dtype=torch.float32)

    _all_branches_single_pass_pooled_kernel[(MM_C,)](
        pool_grad,
        add0,
        add1,
        add2,
        arg511_1.contiguous(),
        arg512_1.contiguous(),
        arg513_1.contiguous(),
        arg209_1.contiguous(),
        arg210_1.contiguous(),
        arg507_1.contiguous(),
        arg508_1.contiguous(),
        arg509_1.contiguous(),
        arg206_1.contiguous(),
        arg207_1.contiguous(),
        arg504_1.contiguous(),
        arg505_1.contiguous(),
        arg506_1.contiguous(),
        arg203_1.contiguous(),
        arg204_1.contiguous(),
        arg495_1.contiguous(),
        arg496_1.contiguous(),
        arg497_1.contiguous(),
        arg196_1.contiguous(),
        arg197_1.contiguous(),
        arg492_1.contiguous(),
        arg493_1.contiguous(),
        arg494_1.contiguous(),
        arg193_1.contiguous(),
        arg194_1.contiguous(),
        arg486_1.contiguous(),
        arg487_1.contiguous(),
        arg488_1.contiguous(),
        arg188_1.contiguous(),
        arg189_1.contiguous(),
        full,
        out_a_tensor,
        out_a_vec,
        out_b_tensor,
        out_b_vec,
        out_c_tensor,
        out_c_vec,
        out_d_tensor,
        out_d_vec,
        out_e_tensor,
        out_e_vec,
        out_f_tensor,
        out_f_vec,
        MM_C_=MM_C,
        H_=H,
        W_=W,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        AVG_SCALE_=AVG_SCALE,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_K=TOTAL_SPATIAL,
        num_warps=SINGLE_PASS_WARPS,
    )

    return (
        out_a_tensor,
        out_a_vec,
        out_b_tensor,
        out_b_vec,
        out_c_tensor,
        out_c_vec,
        out_d_tensor,
        out_d_vec,
        out_e_tensor,
        out_e_vec,
        out_f_tensor,
        out_f_vec,
    )


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_bench(rep: int, warmup: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_full(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_full(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_full split-K pooled-add six-branch BN backward: {oracle_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
