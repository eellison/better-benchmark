"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 194e25f4f7a2
Shape hash: f8ed7d89
"""
_shapes_config = "(T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([512, 480, 14, 14], f32, stride=(94080, 1, 6720, 480)), T([512, 480, 1, 1], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_280: "f32[512, 480, 14, 14]", cat_18: "f32[512, 480, 14, 14]", convolution_53: "f32[512, 480, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[512, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_280, cat_18);  getitem_280 = cat_18 = None
        sum_dim_int_list: "f32[512, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[512, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_53, -3.0)
        lt_scalar: "b8[512, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_53, 3.0);  convolution_53 = None
        bitwise_and_tensor: "b8[512, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[512, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full_default);  bitwise_and_tensor = mul_tensor_1 = full_default = None
        return where_self



def make_inputs():
    return [
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 480, 14, 14], [94080, 1, 6720, 480]),  # getitem_280
    torch.randn(48168960, dtype=torch.float32, device='cuda').as_strided([512, 480, 14, 14], [94080, 1, 6720, 480]),  # cat_18
    torch.randn([512, 480, 1, 1], dtype=torch.float32, device='cuda'),
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
