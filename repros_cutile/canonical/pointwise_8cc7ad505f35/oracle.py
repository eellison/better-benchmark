"""cuTile port of pointwise_8cc7ad505f35: token-mask fanout (Longformer/Roberta)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _token_mask_fanout_kernel(
    x_ptr,        # i64 [numel]
    ne_i32_ptr,   # i32 [numel]
    valid_ptr,    # bool [numel]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    one = ct.full(shape=(BLOCK,), fill_value=1, dtype=ct.int64)
    zero = ct.full(shape=(BLOCK,), fill_value=0, dtype=ct.int64)
    vocab_max = ct.full(shape=(BLOCK,), fill_value=50265, dtype=ct.int64)
    non_pad = x != one
    valid = (x >= zero) & (x < vocab_max) & non_pad
    ct.store(ne_i32_ptr, index=(pid,), tile=ct.astype(non_pad, ct.int32))
    ct.store(valid_ptr, index=(pid,), tile=valid)


@oracle_impl(hardware="B200", point="0105f520", BLOCK=1024)
@oracle_impl(hardware="B200", point="26cc4258", BLOCK=512)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    out_i32 = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.int32,
    )
    valid_base = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bool,
    )
    numel = x.numel()
    x_flat = x.view(numel)
    out_flat = out_i32.view(numel)
    valid_flat = valid_base.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _token_mask_fanout_kernel,
        (x_flat, out_flat, valid_flat, BLOCK),
    )
    return out_i32, valid_base.unsqueeze(-1)
