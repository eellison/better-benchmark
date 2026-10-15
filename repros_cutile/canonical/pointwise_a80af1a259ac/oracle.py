"""cuTile port of pointwise_a80af1a259ac: int64 `!=1` cast to int32 pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _ne_one_i64_to_i32_kernel(
    in_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.load(in_ptr, index=(pid,), shape=(BLOCK,))
    one = ct.full(shape=(BLOCK,), fill_value=1, dtype=ct.int64)
    out_values = ct.astype(values != one, ct.int32)
    ct.store(out_ptr, index=(pid,), tile=out_values)


@oracle_impl(hardware="B200", point="0105f520", BLOCK=1024)
@oracle_impl(hardware="B200", point="3d748156", BLOCK=1024)
@oracle_impl(hardware="B200", point="26cc4258", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (arg0_1,) = inputs
    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.int32,
    )
    n = arg0_1.numel()
    in_flat = arg0_1.reshape(n)
    out_flat = out.view(n)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n // BLOCK, 1, 1),
        _ne_one_i64_to_i32_kernel,
        (in_flat, out_flat, BLOCK),
    )
    return out
