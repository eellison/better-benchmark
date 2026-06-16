"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16-to-fp32 GPT-Neo attention-key layout materialization by writing the fresh contiguous `[B*H, D, S]` clone/view output directly from the contiguous `[B*S, H*D]` projection, whereas Inductor lowers the captured view/view/permute/cast/permute/expand/clone/view chain as generic pointwise layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize the double-permute attention head split as a dedicated transpose-copy template with the dtype conversion folded into the materialization; the fix is NEW_PATTERN: add a guarded head-layout materialization lowering for `view(B,S,H,D).permute(0,2,1,3).to(f32).permute(0,1,3,2).clone().view(B*H,D,S)`."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _key_layout_materialization_kernel(
    input_ptr,
    output_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    s_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]

    batch_head = rows // D
    d_offsets = rows - batch_head * D
    batch = batch_head // H
    head = batch_head - batch * H
    mask = (rows < N_ROWS) & (s_offsets < S)
    values = tl.load(
        input_ptr + (batch * S + s_offsets) * (H * D) + head * D + d_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + rows * S + s_offsets, values, mask=mask)


# af0c9f46: (T([4096, 2048], bf16), S([32,128,2048]), S([32,128,16,128]), S([32,16,128,128]), S([512,128,128]))
@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_ROWS=64, BLOCK_S=128, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_S, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    batch = int(_shape_param_2[0])
    heads = int(_shape_param_2[1])
    head_dim = int(_shape_param_2[2])
    seq = int(_shape_param_2[3])
    output = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3),
        (head_dim * seq, seq, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    n_rows = batch * heads * head_dim
    grid = (triton.cdiv(n_rows, BLOCK_ROWS), triton.cdiv(seq, BLOCK_S))
    _key_layout_materialization_kernel[grid](
        arg0_1,
        output,
        S=seq,
        H=heads,
        D=head_dim,
        N_ROWS=n_rows,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=3,
    )
    return output
