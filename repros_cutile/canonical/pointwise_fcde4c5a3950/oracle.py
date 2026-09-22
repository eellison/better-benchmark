"""cuTile port of pointwise_fcde4c5a3950: T5 attention-mask metadata producer.

Since `iota(1024) >= 0` is tautologically true, `where(ge, lift, scalar_tensor)`
is just `lift_fresh_copy` broadcast to `[8, 1, 1024, 1024]`. cuTile does the
bf16 broadcast-fill for the large where output; metadata outputs are cheap
torch ops.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1024
BATCH = 8


@ct.kernel
def _fill_scalar_bf16_kernel(
    scalar_ptr,    # bf16 [1] (scalar viewed as (1,))
    where_ptr,     # bf16 flat  [BATCH*SEQ*SEQ]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_tile = ct.load(scalar_ptr, index=(0,), shape=(1,))
    x_scalar = ct.reshape(x_tile, ())
    # Fill BLOCK-sized tile with the scalar value.
    zeros = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    tile = zeros + ct.astype(x_scalar, ct.float32)
    ct.store(where_ptr, index=(pid,), tile=ct.astype(tile, ct.bfloat16))


@oracle_impl(hardware="B200", point="2a837a19")
def oracle_forward(inputs):
    arg0_1, _shape_param_0 = inputs
    device = arg0_1.device
    dtype = arg0_1.dtype

    iota = torch.empty_strided((SEQ,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((SEQ,), (1,), device=device, dtype=torch.int64)
    unsqueeze = add.as_strided((1, SEQ), (SEQ, 1))
    unsqueeze_1 = add.as_strided((1, 1, SEQ), (SEQ, SEQ, 1))
    unsqueeze_2 = add.as_strided((1, 1, SEQ, 1), (SEQ, SEQ, 1, 1))
    ge = torch.empty_strided((1, 1, SEQ, 1), (SEQ, SEQ, 1, 1), device=device, dtype=torch.bool)
    expand = ge.as_strided((BATCH, 1, SEQ, SEQ), (0, SEQ, 1, 0))
    lift_fresh_copy = torch.empty_strided((), (), device=device, dtype=dtype)
    scalar_tensor = torch.empty_strided((), (), device=device, dtype=dtype)
    where = torch.empty_strided(
        (BATCH, 1, SEQ, SEQ),
        (SEQ * SEQ, SEQ * SEQ, SEQ, 1),
        device=device,
        dtype=dtype,
    )

    torch.arange(SEQ, out=iota)
    add.copy_(iota)
    ge.fill_(True)
    lift_fresh_copy.copy_(arg0_1)
    scalar_tensor.fill_(-3.3895313892515355e38)

    BLOCK = 4096
    total = BATCH * SEQ * SEQ
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // BLOCK, 1, 1),
        _fill_scalar_bf16_kernel,
        (arg0_1.view(1), where.view(-1), BLOCK),
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
