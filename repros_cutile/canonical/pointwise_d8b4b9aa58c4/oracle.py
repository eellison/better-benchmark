"""cuTile port of pointwise_d8b4b9aa58c4 (NEW_PATTERN): layout materialization
of `[288,64,512]` -> `[9437184]` via permute (source[o,c,s] -> output[o*512+s, c])."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUTER = 288  # 96 * 3
INNER = 64
SEQ = 512
Y_TOTAL = OUTER * SEQ  # 147456
X_TOTAL = INNER


@ct.kernel
def _longformer_layout_kernel(
    src,        # (OUTER, INNER, SEQ) bf16
    dst,        # (Y_TOTAL, INNER) bf16
    YBLOCK: ct.Constant[int],
    XBLOCK: ct.Constant[int],
):
    y_block = ct.bid(0)
    x_block = ct.bid(1)
    y = y_block * YBLOCK + ct.arange(YBLOCK, dtype=ct.int32)
    x = x_block * XBLOCK + ct.arange(XBLOCK, dtype=ct.int32)

    seq = y % 512
    outer = y // 512

    outer_2d = ct.astype(ct.reshape(outer, (YBLOCK, 1)), ct.int64)
    seq_2d = ct.astype(ct.reshape(seq, (YBLOCK, 1)), ct.int64)
    x_2d = ct.astype(ct.reshape(x, (1, XBLOCK)), ct.int64)
    outer_b = outer_2d + ct.zeros(shape=(YBLOCK, XBLOCK), dtype=ct.int64)
    seq_b = seq_2d + ct.zeros(shape=(YBLOCK, XBLOCK), dtype=ct.int64)
    x_b = x_2d + ct.zeros(shape=(YBLOCK, XBLOCK), dtype=ct.int64)
    values = ct.gather(src, (outer_b, x_b, seq_b))
    ct.store(dst, index=(y_block, x_block), tile=values)


@oracle_impl(hardware="B200", point="8c9b0625", YBLOCK=128, XBLOCK=64)
def oracle_forward(inputs, *, YBLOCK: int, XBLOCK: int):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    output_numel = int(_shape_param_1[0])

    output = torch.empty_strided(
        (output_numel,),
        (1,),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # View output as (Y_TOTAL, INNER) contiguous for tile store
    output_2d = output.view(Y_TOTAL, X_TOTAL)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(Y_TOTAL, YBLOCK), ct.cdiv(X_TOTAL, XBLOCK), 1),
        _longformer_layout_kernel,
        (arg0_1, output_2d, YBLOCK, XBLOCK),
    )
    return output
