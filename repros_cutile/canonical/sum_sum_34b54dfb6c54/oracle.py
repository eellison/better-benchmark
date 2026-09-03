"""cuTile port of sum_sum_34b54dfb6c54: ALBERT LN-backward row template.

Ports the Triton `_albert_row_kernel` — for each row of `[ROWS=4096, HIDDEN=4096]`:
  converted = arg0.to(f32)
  weighted = converted * scale
  sum_weighted, sum_weighted_normed = reductions across HIDDEN
  out = row_scale * (weighted * HIDDEN - sum_weighted - normed * sum_weighted_normed)
Outputs: (converted f32, out f32, out.bf16 [ROWS,HIDDEN], out.bf16.permute(1,0)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
HIDDEN = 4096


@ct.kernel
def _albert_row_kernel(
    arg0_ptr,        # bf16 [ROWS, HIDDEN]
    scale_ptr,       # f32  [HIDDEN]
    normed_ptr,      # f32  [ROWS, HIDDEN]
    row_scale_ptr,   # f32  [ROWS]
    convert_out_ptr, # f32  [ROWS, HIDDEN]
    f32_out_ptr,     # f32  [ROWS, HIDDEN]
    bf16_out_ptr,    # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    converted = ct.astype(
        ct.load(arg0_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    scale = ct.reshape(
        ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,)), (1, BLOCK_H)
    )
    normed = ct.load(normed_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weighted = converted * scale

    sum_weighted = ct.sum(weighted, axis=1, keepdims=True)  # (1, 1)
    weighted_normed = weighted * normed
    sum_weighted_normed = ct.sum(weighted_normed, axis=1, keepdims=True)

    row_scale = ct.load(row_scale_ptr, index=(row,), shape=(1,))
    row_scale_2d = ct.reshape(row_scale, (1, 1))

    scaled_weighted = weighted * float(HIDDEN_C)
    centered = scaled_weighted - sum_weighted
    correction = normed * sum_weighted_normed
    out = row_scale_2d * (centered - correction)

    ct.store(convert_out_ptr, index=(row, 0), tile=converted)
    ct.store(f32_out_ptr, index=(row, 0), tile=out)
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="77702286", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs

    convert_out = torch.empty(
        (8, 512, HIDDEN), device=arg0_1.device, dtype=torch.float32
    )
    f32_out = torch.empty(
        (8, 512, HIDDEN), device=arg0_1.device, dtype=torch.float32
    )
    bf16_out = torch.empty(
        (ROWS, HIDDEN), device=arg0_1.device, dtype=torch.bfloat16
    )

    # View arg0 as [ROWS, HIDDEN] and views as [ROWS, HIDDEN].
    arg0_2d = arg0_1.view(ROWS, HIDDEN)
    arg2_2d = arg2_1.view(ROWS, HIDDEN)
    arg3_1d = arg3_1.view(ROWS)
    convert_out_2d = convert_out.view(ROWS, HIDDEN)
    f32_out_2d = f32_out.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _albert_row_kernel,
        (
            arg0_2d,
            arg1_1,
            arg2_2d,
            arg3_1d,
            convert_out_2d,
            f32_out_2d,
            bf16_out,
            HIDDEN,
            BLOCK_H,
        ),
    )
    return convert_out, f32_out, bf16_out, bf16_out.permute(1, 0)
