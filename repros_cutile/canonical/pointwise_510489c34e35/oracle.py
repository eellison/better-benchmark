"""cuTile port of pointwise_510489c34e35: Visformer QKV layout split.

Reads strided channels-last `[batch, 3*heads*d_model, h, w]` bf16 source (which
holds Q, K, V concatenated along the channel axis) and writes three contiguous
`(batch, heads, positions, d_model)` bf16 clones.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _qkv_clone_kernel(
    q_view,   # (B, H, P, D) bf16
    k_view,   # (B, H, P, D) bf16
    v_view,   # (B, H, P, D) bf16
    q_out,    # (B, H, P, D) bf16
    k_out,    # (B, H, P, D) bf16
    v_out,    # (B, H, P, D) bf16
    BLOCK_P: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    b_h = ct.bid(0)
    p_block = ct.bid(1)
    d_block = ct.bid(2)

    # We need to convert linear b_h to (batch, head), but grid supports 3D
    # so treat b_h as combined batch*head; both q/k/v views share the same
    # (B*H, P, D) logical shape.
    q_t = ct.load(q_view, index=(b_h, p_block, d_block), shape=(1, BLOCK_P, BLOCK_D))
    k_t = ct.load(k_view, index=(b_h, p_block, d_block), shape=(1, BLOCK_P, BLOCK_D))
    v_t = ct.load(v_view, index=(b_h, p_block, d_block), shape=(1, BLOCK_P, BLOCK_D))

    ct.store(q_out, index=(b_h, p_block, d_block), tile=q_t)
    ct.store(k_out, index=(b_h, p_block, d_block), tile=k_t)
    ct.store(v_out, index=(b_h, p_block, d_block), tile=v_t)


@oracle_impl(hardware="B200", point="9cb825ed", BLOCK_P=32, BLOCK_D=64)
@oracle_impl(hardware="B200", point="97e22389", BLOCK_P=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int):
    arg0_1, shape0 = inputs
    batch, _qkv, heads, d_model, positions = (int(dim) for dim in shape0)
    out_shape = (batch, heads, positions, d_model)

    # arg0_1 is [batch, 3*heads*d_model, H, W] channels-last (stride like
    # (C*H*W, 1, W*C, C)); the repro reshapes it to
    #   .view(batch, 3, heads, d_model, positions=H*W).unbind(1) → (Q,K,V) with
    #   shape (batch, heads, d_model, positions), then .permute(0,1,3,2) →
    #   (batch, heads, positions, d_model), then .clone() to contiguous.
    #
    # We construct the same views via as_strided so each Q/K/V's logical
    # (B, H, P, D) matches the (B*H, P, D) tile layout the kernel expects.
    total_c = 3 * heads * d_model
    ch_stride = int(arg0_1.stride(1))       # 1
    pos_stride = int(arg0_1.stride(3))      # heads*3*d_model (or channels)
    batch_stride = int(arg0_1.stride(0))
    # Build (B, 3, H, D, P) view
    src_5d = torch.as_strided(
        arg0_1,
        (batch, 3, heads, d_model, positions),
        (
            batch_stride,
            heads * d_model * ch_stride,
            d_model * ch_stride,
            ch_stride,
            pos_stride,
        ),
    )
    # Then permute to (B, 3, H, P, D) so we have (batch, 3, heads, positions, d_model)
    src_5d = src_5d.permute(0, 1, 2, 4, 3)  # -> (B, 3, H, P, D)
    # Each Q/K/V is src_5d[:, i]; the logical (B, H, P, D) shape.
    q_view = src_5d[:, 0].reshape(batch * heads, positions, d_model)
    k_view = src_5d[:, 1].reshape(batch * heads, positions, d_model)
    v_view = src_5d[:, 2].reshape(batch * heads, positions, d_model)

    q_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)
    k_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)
    v_out = torch.empty(out_shape, device=arg0_1.device, dtype=arg0_1.dtype)
    q_bh = q_out.view(batch * heads, positions, d_model)
    k_bh = k_out.view(batch * heads, positions, d_model)
    v_bh = v_out.view(batch * heads, positions, d_model)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch * heads, (positions + BLOCK_P - 1) // BLOCK_P, (d_model + BLOCK_D - 1) // BLOCK_D),
        _qkv_clone_kernel,
        (q_view, k_view, v_view, q_bh, k_bh, v_bh, BLOCK_P, BLOCK_D),
    )
    return q_out, k_out, v_out
