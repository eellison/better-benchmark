"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 194e25f4f7a2
Shape hash: ad218849
"""
_shapes_config = "(T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([512, 960, 1, 1], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_199: "f32[512, 960, 7, 7]", cat_26: "f32[512, 960, 7, 7]", convolution_80: "f32[512, 960, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_199, cat_26);  getitem_199 = cat_26 = None
        sum_dim_int_list: "f32[512, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[512, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_80, -3.0)
        lt_scalar: "b8[512, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_80, 3.0);  convolution_80 = None
        bitwise_and_tensor: "b8[512, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[512, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full_default);  bitwise_and_tensor = mul_tensor_1 = full_default = None
        return where_self



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # getitem_199
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # cat_26
    torch.randn([512, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
