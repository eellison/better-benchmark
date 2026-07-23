"""cuTile port of amax_sum_7932ff4134ad: T5 bidirectional relative-position
attention softmax. The bucket table is precomputed in torch (add_4 side output),
then a cuTile row softmax kernel handles amax/exp/sum/div per row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _row_softmax_kernel(
    x_ptr,           # bf16 [rows, seq_len]  (view + bias baked in bf16)
    out_ptr,         # bf16 [rows, seq_len]
    seq_len: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    x = ct.astype(x_bf16, ct.float32)
    row_max = ct.max(x)
    shifted = x - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs = numer * (1.0 / denom)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(probs, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="ea4c0a34", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, shape0, _shape1, _shape2, _shape3, shape4 = inputs
    del _shape1, _shape2, _shape3

    full_shape = _shape_tuple(shape0)     # [8, 8, 1024, 1024]
    out_shape = _shape_tuple(shape4)      # [64, 1024, 1024]
    batch = int(full_shape[0])
    heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])

    # ---- Compute relative-position bucket + gather bias (add_4) via torch ----
    device = arg0_1.device
    q_pos = torch.arange(q_len, device=device, dtype=torch.int64).unsqueeze(1)  # [q, 1]
    k_pos = torch.arange(k_len, device=device, dtype=torch.int64).unsqueeze(0)  # [1, k]
    rel = k_pos - q_pos  # [q, k]
    gt_zero = (rel > 0).to(torch.int64) * 16
    abs_rel = rel.abs()
    lt_8 = abs_rel < 8

    div = abs_rel.to(torch.float32) / 8.0
    log = torch.log(div)
    div_1 = log / 2.772588722239781
    mul_1 = div_1 * 8
    ct_2 = mul_1.to(torch.int64) + 8
    full15 = torch.full((q_len, k_len), 15, dtype=torch.int64, device=device)
    minimum = torch.minimum(ct_2, full15)
    where = torch.where(lt_8, abs_rel, minimum)
    bucket = gt_zero + where  # i64 [q, k]

    # Gather bias — arg1_1 is bf16[32, 8]. embedding => bf16[q, k, 8]
    embedded = torch.nn.functional.embedding(bucket, arg1_1)  # bf16[q, k, 8]
    # permute -> bf16[8, q, k]
    permuted = embedded.permute(2, 0, 1).contiguous()  # bf16[8, q, k]
    # unsqueeze -> bf16[1, 8, q, k]
    unsq = permuted.unsqueeze(0)
    full_zero = torch.zeros((heads, 1, q_len, k_len), dtype=torch.bfloat16, device=device)
    add_4 = unsq + full_zero  # bf16[8, 8, q, k]  (broadcasts)

    # ---- Softmax input: raw view + add_4 bias, in bf16 ----
    view_raw = arg0_1.view(batch, heads, q_len, k_len)
    scores_bf16 = view_raw + add_4  # bf16 add in torch

    rows = batch * heads * q_len
    scores_2d = scores_bf16.contiguous().view(rows, k_len)
    out_2d = torch.empty_strided(
        (rows, k_len),
        (k_len, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _row_softmax_kernel,
        (scores_2d, out_2d, k_len, BLOCK_N),
    )

    # Reshape output back to expected strided view [64, 1024, 1024]
    out = torch.empty_strided(
        out_shape,
        (q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out.copy_(out_2d.view(out_shape))

    return add_4, out
