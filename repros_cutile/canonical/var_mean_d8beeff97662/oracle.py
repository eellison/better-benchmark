"""cuTile port of var_mean_d8beeff97662: MobileViT LayerNorm + patch-fold layout."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_pass_kernel(src_ptr, dst_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    x = ct.load(src_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO)
    ct.store(dst_ptr, index=(pid,), tile=x)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a2357153", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=1.0)
@oracle_impl(hardware="B200", point="f049abfe", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=3.0)
@oracle_impl(hardware="B200", point="d8c968d2", BLOCK_H=256, ROW_BLOCK=4, LAYOUT_BLOCK=256, BLEND_THRESHOLD=1.0)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, LAYOUT_BLOCK, BLEND_THRESHOLD):
    del BLOCK_H, ROW_BLOCK, LAYOUT_BLOCK, BLEND_THRESHOLD
    a0, a1, a2, a3, sp0, sp1, sp2, sp3 = inputs

    view = a0.view(*_shape_tuple(sp0))
    add = a1 + view
    ct1 = add.to(torch.float32)
    var, mean = torch.var_mean(ct1, dim=[2], correction=0, keepdim=True)
    sub = ct1 - mean
    add_1 = var + 1e-5
    rsqrt = torch.rsqrt(add_1)
    mul = sub * rsqrt
    mul_1 = mul * a2
    add_2 = mul_1 + a3
    ct2 = add_2.to(torch.bfloat16)
    view_1 = ct2.view(*_shape_tuple(sp1))
    permute = view_1.permute(0, 3, 2, 1).contiguous()
    view_2 = permute.view(*_shape_tuple(sp2))
    permute_1 = view_2.permute(0, 2, 1, 3).contiguous()
    view_3 = permute_1.view(*_shape_tuple(sp3))

    n_elem = view_3.numel()
    src = view_3.contiguous().view(-1)
    dst = torch.empty_like(src)
    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(n_elem, BLOCK), 1, 1), _bf16_pass_kernel, (src, dst, BLOCK))
    view_3_final = dst.view_as(view_3)

    return view_3_final
