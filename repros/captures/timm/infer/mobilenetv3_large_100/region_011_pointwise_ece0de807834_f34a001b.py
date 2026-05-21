"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: ece0de807834
Shape hash: f34a001b
"""
_shapes_config = "(T([672], f32), T([512, 672, 14, 14], f32, stride=(131712, 1, 9408, 672)), T([672], f32), T([672], f32), T([672], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg202_1: "f32[672]", convolution_46: "f32[512, 672, 14, 14]", arg203_1: "f32[672]", arg204_1: "f32[672]", arg205_1: "f32[672]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_default_1);  convolution_46 = unsqueeze_default_1 = None
        add_tensor: "f32[672]" = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_default: "f32[672]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_default_5: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_default_7: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_2: "f32[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[512, 672, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_3: "f32[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[512, 672, 14, 14]" = torch.ops.aten.div.Tensor(mul_tensor_3, 6);  mul_tensor_3 = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn(67436544, dtype=torch.float32, device='cuda').as_strided([512, 672, 14, 14], [131712, 1, 9408, 672]),  # convolution_46
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
