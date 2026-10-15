"""cuTile port of pointwise_b1940cc2c2d3: DeBERTaV2 constant mask fanout — write 1.0/True to 24 bool outputs + a float 'mul' output + a full."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_true_bool_kernel(out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    ct.store(out_ptr, index=(pid,), tile=ct.full(shape=(BLOCK,), fill_value=True, dtype=ct.bool_))


@ct.kernel
def _fill_one_f32_kernel(out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    ct.store(out_ptr, index=(pid,), tile=ct.full(shape=(BLOCK,), fill_value=1.0, dtype=ct.float32))


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (shape_param,) = inputs
    batch = int(shape_param[0])
    seq = int(shape_param[1])
    device = torch.device("cuda", 0)

    full = torch.empty_strided((batch, seq), (seq, 1), device=device, dtype=torch.float32)
    unsqueeze = torch.as_strided(full, (batch, seq, 1), (seq, 1, 1))
    unsqueeze_1 = torch.as_strided(full, (batch, 1, seq), (seq, seq, 1))
    unsqueeze_2 = torch.as_strided(full, (batch, 1, 1, seq), (seq, seq, seq, 1))
    squeeze = torch.as_strided(full, (batch, 1, seq), (seq, seq, 1))
    unsqueeze_3 = torch.as_strided(full, (batch, 1, seq, 1), (seq, seq, 1, 1))

    big_shape = (batch, 1, seq, seq)
    big_stride = (seq * seq, seq * seq, seq, 1)
    mul = torch.empty_strided(big_shape, big_stride, device=device, dtype=torch.float32)
    bool_outputs = tuple(
        torch.empty_strided(big_shape, big_stride, device=device, dtype=torch.bool)
        for _ in range(24)
    )

    total = batch * seq * seq
    full_total = batch * seq

    stream = torch.cuda.current_stream()

    # Fill the small 'full' tensor with 1.0 (numel=4096 -> divides BLOCK=1024)
    full_flat = full.view(full_total)
    ct.launch(
        stream,
        (ct.cdiv(full_total, BLOCK), 1, 1),
        _fill_one_f32_kernel,
        (full_flat, BLOCK),
    )

    # Fill 'mul' (float32, all 1.0)
    mul_flat = mul.view(total)
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _fill_one_f32_kernel,
        (mul_flat, BLOCK),
    )

    # Fill each bool output with True
    for out in bool_outputs:
        out_flat = out.view(total)
        ct.launch(
            stream,
            (ct.cdiv(total, BLOCK), 1, 1),
            _fill_true_bool_kernel,
            (out_flat, BLOCK),
        )

    return (
        full,
        unsqueeze,
        unsqueeze_1,
        unsqueeze_2,
        squeeze,
        unsqueeze_3,
        mul,
        *bool_outputs,
    )
