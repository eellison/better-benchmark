"""cuTile port of pointwise_c3d0c3208b3d: MT5 attention-mask producer.

The Repro just produces a bunch of constant/simple-fill tensors:
- iota, add: [0..127] i64
- ge: all-true bool
- lift/scalar: bf16 scalars
- where: [B, 1, 128, 128] filled with x

We fill the where tensor via a small cuTile kernel and use torch ops for the
metadata/small tensors.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_where_kernel(
    x_ptr,        # bf16 scalar (as a length-1 array)
    where_ptr,    # bf16 flat [TOTAL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(0,), shape=(1,))
    # broadcast scalar to a tile of size BLOCK
    x_tile = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16) + x
    ct.store(where_ptr, index=(pid,), tile=x_tile)


@oracle_impl(hardware="B200", point="2a837a19", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, shape_param = inputs
    device = arg0_1.device
    dtype = arg0_1.dtype
    batch = int(shape_param[0])
    seq_len = int(shape_param[2])

    iota = torch.arange(seq_len, device=device, dtype=torch.int64)
    add = iota  # + 0
    unsqueeze = torch.as_strided(add, (1, seq_len), (seq_len, 1))
    unsqueeze_1 = torch.as_strided(add, (1, 1, seq_len), (seq_len, seq_len, 1))
    unsqueeze_2 = torch.as_strided(add, (1, 1, seq_len, 1), (seq_len, seq_len, 1, 1))

    ge = torch.ones((1, 1, seq_len, 1), device=device, dtype=torch.bool)
    expand = torch.as_strided(ge, (batch, 1, seq_len, seq_len), (0, seq_len, 1, 0))

    lift_fresh_copy = arg0_1.clone()
    scalar_tensor = torch.tensor(-3.3895313892515355e38, device=device, dtype=dtype)

    where_shape = (batch, 1, seq_len, seq_len)
    where = torch.empty_strided(
        where_shape,
        (seq_len * seq_len, seq_len * seq_len, seq_len, 1),
        device=device,
        dtype=dtype,
    )
    total = where.numel()

    # Ensure arg0_1 is a length-1 array for the kernel (0-dim tiles not supported).
    x_flat = arg0_1.view(1)
    where_flat = where.view(-1)

    for cand in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if total % cand == 0:
            block = cand
            break

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, block), 1, 1),
        _fill_where_kernel,
        (x_flat, where_flat, block),
    )

    return (
        iota,
        add,
        unsqueeze,
        unsqueeze_1,
        unsqueeze_2,
        ge,
        expand,
        lift_fresh_copy,
        scalar_tensor,
        where,
    )
