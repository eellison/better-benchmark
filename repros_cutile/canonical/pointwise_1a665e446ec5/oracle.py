"""cuTile port of pointwise_1a665e446ec5: Mistral rotary-position + grouped-KV.

Uses torch for the layout ops (iota, unsqueeze, permute, expand, cat) and a
cuTile kernel for the RoPE rotation pointwise (mul_2 = permute * cos_bf16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _rope_mul_kernel(
    x_ptr,       # bf16 [rows, cols]
    cos_ptr,     # bf16 [rows, cols]
    out_ptr,     # bf16 [rows, cols]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    c = ct.load(cos_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=x * c)


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="cc5826e9", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, shape2, shape3, shape4, shape5, shape6, shape7, shape8, shape9 = inputs
    device = arg0_1.device

    view = arg0_1.view(_shape(shape0))
    view_1 = view.view(_shape(shape1))
    permute = view_1.permute(0, 2, 1, 3)  # [1, 32, 1000, 128]

    # Build the freq table: expand_1: f32[1, 64, 1] = arg1_1 broadcast
    unsqueeze = arg1_1.unsqueeze(0)  # [1, 64]
    unsqueeze_1 = unsqueeze.unsqueeze(2)  # [1, 64, 1]
    convert_element_type = unsqueeze_1.to(torch.float32)
    expand_1 = convert_element_type.expand(_shape(shape2))
    iota = torch.arange(1000, device=device, dtype=torch.int64)
    add_val = iota + 0
    unsqueeze_2 = add_val.unsqueeze(0)  # [1, 1000]
    unsqueeze_3 = unsqueeze_2.unsqueeze(1)  # [1, 1, 1000]
    convert_element_type_1 = unsqueeze_3.to(torch.float32)
    expand_2 = convert_element_type_1.expand(_shape(shape3))
    mul_0 = expand_1 * expand_2  # [1, 64, 1000]
    permute_1 = mul_0.permute(0, 2, 1)  # [1, 1000, 64]
    unsqueeze_4 = permute_1.unsqueeze(2)  # [1, 1000, 1, 64]
    expand_3 = unsqueeze_4.expand(_shape(shape4))
    clone = expand_3.contiguous()
    view_2 = clone.view(_shape(shape5))  # [1, 1000, 128]

    cos_v = view_2.cos() * 1.0
    convert_element_type_2 = cos_v.to(torch.bfloat16)
    unsqueeze_5 = convert_element_type_2.unsqueeze(1)  # [1, 1, 1000, 128]

    # cuTile: mul_2 = permute * unsqueeze_5 (broadcast)
    mul_2_shape = tuple(permute.shape)  # [1, 32, 1000, 128]
    numel = 1
    for d in mul_2_shape:
        numel *= int(d)
    permute_contig = permute.contiguous()
    cos_broadcast = unsqueeze_5.expand(mul_2_shape).contiguous()
    mul_2 = torch.empty(mul_2_shape, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _rope_mul_kernel,
              (permute_contig.view(numel), cos_broadcast.view(numel),
               mul_2.view(numel), BLOCK))

    slice_1 = permute[..., 64:]  # last half
    neg = -slice_1
    slice_2 = permute[..., :64]  # first half
    cat = torch.cat([neg, slice_2], dim=-1)

    sin_v = view_2.sin() * 1.0
    convert_element_type_3 = sin_v.to(torch.bfloat16)
    unsqueeze_6 = convert_element_type_3.unsqueeze(1)

    mul_4 = cat * unsqueeze_6
    add_1 = mul_2 + mul_4

    view_3 = arg2_1.view(_shape(shape6))
    view_4 = view_3.view(_shape(shape7))
    permute_2 = view_4.permute(0, 2, 1, 3)
    mul_5 = permute_2 * unsqueeze_5
    slice_3 = permute_2[..., 64:]
    neg_1 = -slice_3
    slice_4 = permute_2[..., :64]
    cat_1 = torch.cat([neg_1, slice_4], dim=-1)
    mul_6 = cat_1 * unsqueeze_6
    add_2 = mul_5 + mul_6

    unsqueeze_7 = add_2.unsqueeze(2)
    expand_4 = unsqueeze_7.expand(_shape(shape8))
    clone_1 = expand_4.contiguous()
    view_5 = clone_1.view(_shape(shape9))

    return convert_element_type_2, convert_element_type_3, add_1, add_2, view_5
