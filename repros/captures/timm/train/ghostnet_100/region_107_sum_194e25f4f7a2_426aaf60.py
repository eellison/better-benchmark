"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 194e25f4f7a2
Shape hash: 426aaf60
"""
_shapes_config = "(T([512, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([512, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([512, 672, 1, 1], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_256: "f32[512, 672, 14, 14]", cat_20: "f32[512, 672, 14, 14]", convolution_61: "f32[512, 672, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_256, cat_20);  getitem_256 = cat_20 = None
        sum_dim_int_list: "f32[512, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2, 3], True);  mul_tensor = None
        gt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_61, -3.0)
        lt_scalar: "b8[512, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_61, 3.0);  convolution_61 = None
        bitwise_and_tensor: "b8[512, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_1: "f32[512, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_1, full_default);  bitwise_and_tensor = mul_tensor_1 = full_default = None
        return where_self



def make_inputs():
    return [
    torch.randn(67436544, dtype=torch.float32, device='cuda').as_strided([512, 672, 14, 14], [131712, 1, 9408, 672]),  # getitem_256
    torch.randn(67436544, dtype=torch.float32, device='cuda').as_strided([512, 672, 14, 14], [131712, 1, 9408, 672]),  # cat_20
    torch.randn([512, 672, 1, 1], dtype=torch.float32, device='cuda'),
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
