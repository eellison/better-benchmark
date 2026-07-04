"""cuTile port of pointwise_bf1f8ed80b12: Visformer Q/K/V layout split.

The whole scope is a pure layout op: view + permute + unbind + contiguous
clones + constant zero pads + permute views. cuTile has no dedicated "gather
from a strided producer" primitive that would give us the same throughput as
Triton's fused kernel, but we can still exercise cuTile with a substantive
"copy + zero-pad" kernel that takes each contiguous QKV plane [rows, positions,
d_model] and writes a [rows, padded_positions, d_model] tile with zeros in the
right-hand pad columns. The strided permute source is materialised via a
single torch contiguous() up-front (the same work Triton's kernel does).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _zero_pad_copy_kernel(
    src_ptr,       # bf16 flat [rows * positions * d_model], contiguous
    dst_ptr,       # bf16 flat [rows * padded_positions * d_model], contiguous
    rows: ct.Constant[int],
    positions: ct.Constant[int],
    padded_positions: ct.Constant[int],
    d_model: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    idx = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    total_out = rows * padded_positions * d_model
    valid = idx < total_out

    d = idx % d_model
    tmp = idx // d_model
    p = tmp % padded_positions
    row = tmp // padded_positions

    within_positions = p < positions
    src_off = (row * positions + p) * d_model + d
    zero_off = ct.zeros((BLOCK,), dtype=ct.int32)
    src_off_safe = ct.where(within_positions, src_off, zero_off)
    vals_bf = ct.gather(src_ptr, (src_off_safe,), mask=valid & within_positions)
    ct.scatter(dst_ptr, (idx,), vals_bf, mask=valid)


def _launch_pad(src_contig, padded_positions, *, block=1024):
    rows, positions, d_model = src_contig.shape
    device = src_contig.device
    dst = torch.zeros(
        (rows, padded_positions, d_model), device=device, dtype=torch.bfloat16
    )
    src_flat = src_contig.contiguous().view(-1)
    dst_flat = dst.view(-1)
    total_out = rows * padded_positions * d_model
    grid = ((total_out + block - 1) // block, 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _zero_pad_copy_kernel,
        (src_flat, dst_flat, rows, positions, padded_positions, d_model, block),
    )
    return dst


@oracle_impl(hardware="B200", point="97e22389", QV_BLOCK=1024, K_YBLOCK=16, K_XBLOCK=64)
def oracle_forward(inputs, *, QV_BLOCK: int, K_YBLOCK: int, K_XBLOCK: int):
    del K_YBLOCK, K_XBLOCK  # unused in cuTile port
    source = inputs[0]
    view_shape = tuple(int(dim) for dim in inputs[1])  # [128, 3, 6, 64, 196]

    # Reshape strided source into logical [B, 3, heads, d_model, positions]
    # via .view (channels-last stride makes this a pure view of the storage).
    view = source.view(*view_shape)
    # permute [1, 0, 2, 4, 3] -> [3, B, heads, positions, d_model]
    permuted = view.permute(1, 0, 2, 4, 3)
    q, k, v = permuted[0], permuted[1], permuted[2]
    # k comes back as [B, heads, positions, d_model]; the repro then permutes it
    # to [B, heads, d_model, positions] via permute_1 = permute(k, [0,1,3,2]).
    k_pt = k.permute(0, 1, 3, 2)  # [B, heads, d_model, positions]

    B, heads, positions, d_model = q.shape
    rows = B * heads
    padded_positions = positions + 4  # from _shape_param_6/9 (0,0,0,4,...)

    # Contiguous [rows, positions, d_model] and [rows, d_model, positions] planes.
    q_contig = q.contiguous().view(rows, positions, d_model)
    v_contig = v.contiguous().view(rows, positions, d_model)
    k_contig = k_pt.contiguous().view(rows, d_model, positions)

    q_pad = _launch_pad(q_contig, padded_positions, block=QV_BLOCK)
    # For K, positions is the last dim; treat as [rows, d_model, positions]
    # and pad the last axis. Use the same kernel by swapping role names:
    # "positions" becomes d_model, "d_model" becomes... no — that changes
    # semantics. Do a dedicated pad by transposing to [rows, positions, d_model]
    # first, padding, then transposing back.
    k_as_pos_last = k_contig.transpose(1, 2).contiguous()  # [rows, positions, d_model]
    k_pad_pos_last = _launch_pad(k_as_pos_last, padded_positions, block=QV_BLOCK)
    # Now we need [rows, d_model, padded_positions]; k_pad_pos_last is
    # [rows, padded_positions, d_model]. transpose(1,2) then contiguous.
    k_pad = k_pad_pos_last.transpose(1, 2).contiguous()

    v_pad = _launch_pad(v_contig, padded_positions, block=QV_BLOCK)

    return (
        q_pad,                        # constant_pad_nd  [768, 200, 64]
        k_pad,                        # constant_pad_nd_1 [768, 64, 200]
        v_pad,                        # constant_pad_nd_2 [768, 200, 64]
        v_contig.permute(0, 2, 1),    # permute_2 (view of v_base) [768, 64, 196]
        q_contig.permute(0, 2, 1),    # permute_3 (view of q_base) [768, 64, 196]
        k_contig.permute(0, 2, 1),    # permute_4 (view of k_base) [768, 196, 64]
    )
