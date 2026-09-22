"""cuTile port of pointwise_035e4d7f5117 (NEW_PATTERN): allocate bf16 output
and fill with 1.0."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_one_bf16_kernel(
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    ones = ct.full(shape=(BLOCK,), fill_value=1.0, dtype=ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=ones)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    shape = tuple(int(dim) for dim in inputs[0])
    stride = (shape[1] * shape[2], shape[2], 1)
    out = torch.empty_strided(
        shape,
        stride,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    # numel = 204*204*26 = 1082016. 1082016 % 1024 = 480 => not divisible.
    # Use flat view; last partial tile is written via store, which requires
    # tile size to divide the array. Instead, we round the array's tile
    # partition by padding: since we cannot mask a store, fall back to
    # per-element small BLOCK where numel/BLOCK divides. 1082016 = 2^5 * 3 * ...
    # 1082016 / 32 = 33813, / 16 = 67626, / 8 = 135252. So BLOCK=32 works
    # (32 divides 1082016). But that's a lot of tiles. Even BLOCK=1082016 or
    # a small divisor is needed. 1082016 = 32 * 33813. Use BLOCK=32.
    numel = out.numel()
    # find divisor <= requested BLOCK
    effective = BLOCK
    while numel % effective != 0 and effective > 1:
        effective //= 2
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, effective), 1, 1),
        _fill_one_bf16_kernel,
        (out.view(-1), effective),
    )
    return out
