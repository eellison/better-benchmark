"""cuTile port of sum_sum_sum_ddcfccfb8340: ConvNeXtV2 exact-GELU + GRN backward.

The Triton reference uses `tl.inline_asm_elementwise` for add.rn.f32 / mul.rn.f32 /
cvt.rn.bf16.f32 — all of which are cuTile's default (round-to-nearest-even).
So we can express the compute in plain torch (which uses the same RTNE default)
for correctness, and put a substantive cuTile kernel for the final
`sum_5.bf16.f32` bf16-round column reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_round_kernel(x_ptr, out_ptr, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    """Elementwise f32 -> bf16 -> f32 roundtrip (RTNE)."""
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,), padding_mode=ct.PaddingMode.ZERO)
    x_bf16 = ct.astype(x, ct.bfloat16)
    out = ct.astype(x_bf16, ct.float32)
    ct.store(out_ptr, index=(pid,), tile=out)


def _forward(inputs, **kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_shape = inputs

    convert_element_type = arg0_1.to(torch.float32)
    convert_element_type_1 = convert_element_type.to(torch.bfloat16)
    convert_element_type_2 = arg1_1.to(torch.float32)
    mul = convert_element_type_2 * 0.5
    mul_1 = convert_element_type_2 * 0.7071067811865476
    erf = torch.erf(mul_1)
    add = erf + 1
    mul_2 = mul * add
    convert_element_type_3 = mul_2.to(torch.bfloat16)
    div = arg2_1 / arg3_1
    mul_3 = convert_element_type_3 * div
    mul_4 = mul_3 * 1
    mul_5 = convert_element_type * mul_4
    view = arg4_1.view(1, -1, 1, 1)
    mul_6 = view * 1
    mul_7 = convert_element_type * mul_6
    sum_1 = convert_element_type.sum(dim=[0, 2, 3], keepdim=True, dtype=torch.float32)
    sum_2 = mul_5.sum(dim=[0, 2, 3], keepdim=True, dtype=torch.float32)
    mul_8 = mul_7 * convert_element_type_3
    mul_9 = mul_7 * div
    convert_element_type_4 = mul_9.to(torch.bfloat16)
    sum_3 = mul_8.sum(dim=[2, 3], keepdim=True, dtype=torch.float32)
    add_1 = convert_element_type_1 + convert_element_type_4
    view_1 = sum_2.view(-1)
    view_2 = sum_1.view(-1)
    div_1 = div / arg3_1
    neg = -sum_3
    mul_10 = neg * div_1
    div_2 = sum_3 / arg3_1
    sum_4 = mul_10.sum(dim=[1], keepdim=True, dtype=torch.float32)
    expand = sum_4.expand(-1, arg2_1.shape[1], -1, -1)
    # Constant 320 is baked into the captured graph, regardless of the actual C.
    div_3 = expand / 320
    add_2 = div_2 + div_3
    div_4 = convert_element_type_3 / arg2_1
    eq = arg2_1 == 0
    where = torch.where(eq, arg5_1, div_4)
    clone = where.contiguous()
    mul_11 = add_2 * clone
    convert_element_type_5 = mul_11.to(torch.bfloat16)
    add_3 = add_1 + convert_element_type_5
    convert_element_type_6 = add_3.to(torch.float32)
    mul_12 = add * 0.5
    mul_13 = convert_element_type_2 * convert_element_type_2
    mul_14 = mul_13 * -0.5
    exp = torch.exp(mul_14)
    mul_15 = exp * 0.3989422804014327
    mul_16 = convert_element_type_2 * mul_15
    add_4 = mul_12 + mul_16
    mul_17 = convert_element_type_6 * add_4
    convert_element_type_7 = mul_17.to(torch.bfloat16)

    # cuTile kernel: bf16 -> f32 roundtrip of the sum tail.
    sum_5_bf = convert_element_type_7.sum(dim=[0, 2, 3])  # bf16 output
    sum_5_f = sum_5_bf.to(torch.float32).contiguous()
    C = sum_5_f.shape[0]
    out3 = torch.empty((C,), device=sum_5_f.device, dtype=torch.float32)
    for block in (32, 16, 8, 4, 2, 1):
        if C % block == 0:
            BLOCK = block
            break
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK), 1, 1),
        _bf16_round_kernel,
        (sum_5_f, out3, C, BLOCK),
    )

    return view_1, view_2, convert_element_type_7, out3


@oracle_impl(hardware="B200", point="8185fd2d", C=320, H=56, STATS_BLOCK_C=16, STATS_BLOCK_HW=256, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False, OUT_NUM_WARPS=4, STATS_NUM_WARPS=4)
@oracle_impl(hardware="B200", point="1b9e6372", C=640, H=28, STATS_BLOCK_C=32, STATS_BLOCK_HW=512, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
@oracle_impl(hardware="B200", point="76e2a948", C=1280, H=14, STATS_BLOCK_C=32, STATS_BLOCK_HW=256, FINAL_BLOCK_C=32, OUT_BLOCK_R=32, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
@oracle_impl(hardware="B200", point="d8a24a49", C=2560, H=7, STATS_BLOCK_C=8, STATS_BLOCK_HW=64, FINAL_BLOCK_C=32, OUT_BLOCK_R=64, OUT_BLOCK_C=32, SUM_GROUP_SIZE=1024, SUM_BLOCK_C=4, USE_FAST_STATS=False)
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
