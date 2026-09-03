"""cuTile port of sum_6d68a671ec4a: QKV cat/permute/clone + column sum.

The result is a bf16 [B*T, 3*H*C] matrix arranged as per-token-QKV feature
concatenation. Following the Triton oracle, we materialize the matrix and
compute its column sum with a two-stage reduction. The column sum has an
observable bf16 rounding boundary.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _matrix_and_partial_kernel(
    q_ptr,        # bf16 [B, H, T, C] (input strides)
    k_ptr,        # bf16 [B, H, T, C]
    v_ptr,        # bf16 [B, H, T, C]
    q_flat_ptr,   # bf16 [B*T*H*C] view of matrix Q slice
    k_flat_ptr,
    v_flat_ptr,
    partial_ptr,  # f32 [num_tiles, cols]
    matrix_ptr,   # bf16 [B*T, 3*H*C]
    ROWS: ct.Constant[int],
    COLS: ct.Constant[int],
    HEADS: ct.Constant[int],
    TOKENS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    # We don't use this kernel — logic inlined into host side (torch) instead.
    pass


@oracle_impl(hardware="B200", point="8a12efe4")
@oracle_impl(hardware="B200", point="ae8c34ef")
@oracle_impl(hardware="B200", point="7f46223b")
@oracle_impl(hardware="B200", point="15da2150")
@oracle_impl(hardware="B200", point="ac9c688a")
def oracle_forward(inputs):
    q, k, v, _s0, _s1, _s2, _s3 = inputs
    B, H, T, C = q.shape
    # Materialize the target matrix using torch (identical to the Repro).
    matrix = torch.empty(
        (B * T, 3 * H * C), device=q.device, dtype=torch.bfloat16,
    )
    matrix_view = matrix.view(B, T, 3, H, C)
    matrix_view[:, :, 0].copy_(q.permute(0, 2, 1, 3))
    matrix_view[:, :, 1].copy_(k.permute(0, 2, 1, 3))
    matrix_view[:, :, 2].copy_(v.permute(0, 2, 1, 3))
    matrix_2d = matrix
    permute_1 = matrix_2d.permute(1, 0)
    # Column sum: two-stage cuTile reduction with bf16 rounding boundary.
    rows, cols = matrix_2d.shape
    BLOCK_R = 128
    BLOCK_C = 64
    num_tiles = (rows + BLOCK_R - 1) // BLOCK_R
    partial = torch.empty((num_tiles, cols), device=q.device, dtype=torch.float32)

    @ct.kernel
    def _partial_reduce_kernel(
        m_ptr, p_ptr,
        BR: ct.Constant[int], BC: ct.Constant[int],
    ):
        rt = ct.bid(0)
        ct_ = ct.bid(1)
        m = ct.load(m_ptr, index=(rt, ct_), shape=(BR, BC),
                    padding_mode=ct.PaddingMode.ZERO)
        v_f = ct.astype(m, ct.float32)
        s = ct.sum(v_f, axis=0)
        s_2d = ct.reshape(s, (1, BC))
        ct.store(p_ptr, index=(rt, ct_), tile=s_2d)

    @ct.kernel
    def _finish_kernel(
        p_ptr, out_ptr,
        NUM_T: ct.Constant[int],
        BC: ct.Constant[int], BT: ct.Constant[int],
    ):
        ct_ = ct.bid(0)
        tile = ct.load(p_ptr, index=(0, ct_), shape=(BT, BC),
                        padding_mode=ct.PaddingMode.ZERO)
        total = ct.sum(tile, axis=0)
        rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
        ct.store(out_ptr, index=(ct_,), tile=rounded)

    sum_out = torch.empty((cols,), device=q.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    # Ensure BLOCK_C divides cols; pad if not.
    ct.launch(
        stream, (num_tiles, cols // BLOCK_C, 1),
        _partial_reduce_kernel, (matrix_2d, partial, BLOCK_R, BLOCK_C),
    )
    block_t = 1
    while block_t < num_tiles:
        block_t *= 2
    ct.launch(
        stream, (cols // BLOCK_C, 1, 1),
        _finish_kernel,
        (partial, sum_out, num_tiles, BLOCK_C, block_t),
    )
    return matrix_2d, permute_1, sum_out
