"""cuTile port of sum_6d8612892024: attention softmax-backward + dropout.

Ports the Triton `_softmax_backward_dropout_kernel`. For each row (of the
`[BATCH * HEADS * QLEN, KLEN]` flattened tensor):
  * load bf16 grad, bool dropout mask, bf16 logits, fp32 row_shift, fp32 row_denom,
    bool zero_row;
  * scale grad by (1.1111111... * dropout_mask), rounded through bf16;
  * reconstruct probs = exp(logits - row_shift) / row_denom, rounded to bf16, with
    all-masked rows zeroed;
  * row_sum = sum(grad * probs);
  * out = (-probs * row_sum + grad*probs), cast to bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_backward_dropout_kernel(
    grad_ptr,          # bf16 [N_ROWS, KLEN]
    dropout_mask_ptr,  # bool [N_ROWS, KLEN]
    logits_ptr,        # bf16 [N_ROWS, KLEN]
    row_shift_ptr,     # f32 [N_ROWS]
    row_denom_ptr,     # f32 [N_ROWS]
    zero_row_ptr,      # bool [N_ROWS]
    out_ptr,           # bf16 [N_ROWS, KLEN]
    KLEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    keep = ct.load(dropout_mask_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    grad = ct.load(grad_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    logits = ct.load(logits_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    row_shift = ct.load(row_shift_ptr, index=(row_block,), shape=(BLOCK_M,))
    row_denom = ct.load(row_denom_ptr, index=(row_block,), shape=(BLOCK_M,))
    zero_row = ct.load(zero_row_ptr, index=(row_block,), shape=(BLOCK_M,))

    dropout_scale_f = ct.full((BLOCK_M, BLOCK_N), DROPOUT_SCALE, dtype=ct.float32)
    zero_f = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    scaled_mask_bf = ct.astype(ct.where(keep, dropout_scale_f, zero_f), ct.bfloat16)

    grad_f = ct.astype(grad, ct.float32)
    scaled_grad_f = grad_f * ct.astype(scaled_mask_bf, ct.float32)
    scaled_grad_bf = ct.astype(scaled_grad_f, ct.bfloat16)
    scaled_grad_r = ct.astype(scaled_grad_bf, ct.float32)

    logits_f = ct.astype(logits, ct.float32)
    row_shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    row_denom_2d = ct.reshape(row_denom, (BLOCK_M, 1))
    numer = ct.exp(logits_f - row_shift_2d)
    probs_f = numer / row_denom_2d
    probs_bf = ct.astype(probs_f, ct.bfloat16)

    zero_row_2d = ct.reshape(zero_row, (BLOCK_M, 1))
    zero_bf_tile = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    probs_bf = ct.where(zero_row_2d, zero_bf_tile, probs_bf)
    probs_r = ct.astype(probs_bf, ct.float32)

    prod = scaled_grad_r * probs_r
    row_sum = ct.sum(prod, axis=1)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    out_f = -probs_r * row_sum_2d + prod
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="4e534079")
def _oracle_forward_electra(inputs, **_kwargs):
    # ElectraForCausalLM (256, 4, 512, 512): bf16 rounding drift in the
    # softmax-backward chain accumulates past the 1e-2 tol without exact
    # PTX rn intrinsics for div/fma.
    raise NotImplementedError(
        "cuTile port unsupported for 4e534079: bf16 softmax-backward drift beyond 1e-2 tol"
    )


@oracle_impl(hardware="B200", point="931c2d63", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="79d7858b", BLOCK_M=1, BLOCK_N=512)
@oracle_impl(hardware="B200", point="5d18732f", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        arg0_1,           # bf16 grad
        arg1_1,           # bool dropout mask
        arg2_1,           # bf16 logits
        arg3_1,           # fp32 row_shift
        arg4_1,           # fp32 row_denom
        arg5_1,           # bool zero_row
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    out_shape = tuple(int(dim) for dim in _shape_param_3)
    k_len = int(arg1_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    grad_2d = arg0_1.view(n_rows, k_len)
    dropout_2d = arg1_1.contiguous().view(n_rows, k_len)
    logits_2d = arg2_1.view(n_rows, k_len)
    row_shift_1d = arg3_1.contiguous().view(n_rows)
    row_denom_1d = arg4_1.contiguous().view(n_rows)
    zero_row_1d = arg5_1.contiguous().view(n_rows)
    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _softmax_backward_dropout_kernel,
        (grad_2d, dropout_2d, logits_2d, row_shift_1d, row_denom_1d, zero_row_1d,
         out_2d, k_len, BLOCK_M, BLOCK_N),
    )
    return out
