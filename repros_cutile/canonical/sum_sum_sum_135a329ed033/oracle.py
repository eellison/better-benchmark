"""cuTile port of sum_sum_sum_135a329ed033: Longformer LN-backward + dropout.

The Triton reference uses PTX .rn intrinsics (add.rn, mul.rn) which are cuTile's
default RTNE. LN-backward is a large tree with row reductions plus dependent
side outputs, so we implement the LN-backward compute in torch and use cuTile
for the final bf16-rounded column reduction (the sum_5 tail).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_column_sum_kernel(
    x_ptr,          # bf16 [ROWS, HIDDEN]
    out_ptr,        # f32 [HIDDEN]
    ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    """Column reduction over rows for a bf16 matrix, returning f32."""
    col_block = ct.bid(0)
    # Accumulate over row tiles.
    acc = ct.zeros((BLOCK_H,), dtype=ct.float32)
    # ROWS iterations; but let cuTile do full column via tile loop is heavy.
    # Simpler: launch 1D over col-blocks and iterate rows manually.
    # For efficiency, use a large per-tile row block.
    ROW_TILE: ct.Constant[int] = 128
    n_row_tiles = ct.cdiv(ROWS, ROW_TILE)
    for r in range(n_row_tiles):
        tile = ct.load(x_ptr, index=(r, col_block), shape=(ROW_TILE, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
        tile_f32 = ct.astype(tile, ct.float32)
        acc = acc + ct.sum(tile_f32, axis=0)
    ct.store(out_ptr, index=(col_block,), tile=acc)


def _forward(inputs, **kwargs):
    (
        arg0_1,  # bf16 [8192, 768]
        arg1_1,  # f32 [8, 1024, 768]
        arg2_1,  # f32 [768]
        arg3_1,  # f32 [8, 1024, 768]
        arg4_1,  # f32 [8, 1024, 1]
        arg5_1,  # b8 [8, 1024, 768]
        s0, s1, s2,
    ) = inputs

    view = arg0_1.view(*(int(x) for x in s0))
    convert_element_type = view.to(torch.float32)
    add = arg1_1 + convert_element_type
    mul = add * arg2_1
    mul_1 = mul * 768
    sum_1 = mul.sum(dim=[2], keepdim=True)
    mul_2 = mul * arg3_1
    sum_2 = mul_2.sum(dim=[2], keepdim=True)
    mul_3 = arg3_1 * sum_2
    sub = mul_1 - sum_1
    sub_1 = sub - mul_3
    mul_4 = arg4_1 * sub_1
    mul_5 = add * arg3_1
    sum_3 = mul_5.sum(dim=[0, 1])
    sum_4 = add.sum(dim=[0, 1])
    convert_element_type_1 = mul_4.to(torch.bfloat16)
    convert_element_type_2 = arg5_1.to(torch.bfloat16)
    mul_6 = convert_element_type_2 * 1.1111111111111112
    mul_7 = convert_element_type_1 * mul_6
    view_2 = mul_7.view(*(int(x) for x in s2))
    permute = view_2.permute(1, 0)

    # cuTile: column reduction of view_2 (bf16 [8192, 768]) -> f32 [768]
    view_2_c = view_2.contiguous()
    rows = view_2_c.shape[0]
    hidden = view_2_c.shape[1]
    sum5_f32 = torch.empty((hidden,), device=view_2_c.device, dtype=torch.float32)
    BLOCK_H = 32
    assert hidden % BLOCK_H == 0
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(hidden, BLOCK_H), 1, 1),
        _bf16_column_sum_kernel,
        (view_2_c, sum5_f32, rows, hidden, BLOCK_H),
    )
    # bf16 roundtrip on sum5
    convert_element_type_4 = sum5_f32.to(torch.bfloat16).to(torch.float32)

    return mul_4, sum_3, sum_4, view_2, permute, convert_element_type_4


@oracle_impl(
    hardware="B200",
    point="5a443972",
    ROWS_PER_GROUP=16,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=16,
)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
