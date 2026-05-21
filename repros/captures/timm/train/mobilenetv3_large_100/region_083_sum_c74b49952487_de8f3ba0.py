"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: c74b49952487
Shape hash: de8f3ba0
"""
_shapes_config = "(T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([1, 960, 1, 1], f32), T([1, 960, 1, 1], f32), T([960], f32), T([960], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([512, 960, 1, 1], f32), T([], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_52: "f32[512, 960, 7, 7]", getitem_81: "f32[1, 960, 1, 1]", rsqrt_40: "f32[1, 960, 1, 1]", primals_270: "f32[960]", primals_271: "f32[960]", getitem_113: "f32[512, 960, 7, 7]", convolution_54: "f32[512, 960, 1, 1]", full_default: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_81);  convolution_52 = getitem_81 = None
        mul_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_40);  sub_tensor = rsqrt_40 = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor, 3)
        clamp_min_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0);  add_tensor_1 = None
        clamp_max_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, clamp_max_default);  add_tensor = clamp_max_default = None
        div_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_2, 6);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_tensor_3: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, div_tensor);  getitem_113 = div_tensor = None
        sum_dim_int_list: "f32[512, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2, 3], True);  mul_tensor_3 = None
        gt_scalar: "b8[512, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_54, -3.0)
        lt_scalar: "b8[512, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_54, 3.0);  convolution_54 = None
        bitwise_and_tensor: "b8[512, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_scalar, lt_scalar);  gt_scalar = lt_scalar = None
        mul_tensor_4: "f32[512, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.16666666666666666);  sum_dim_int_list = None
        where_self: "f32[512, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_tensor, mul_tensor_4, full_default);  bitwise_and_tensor = mul_tensor_4 = full_default = None
        return where_self



def make_inputs():
    return [
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # convolution_52
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # getitem_113
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
