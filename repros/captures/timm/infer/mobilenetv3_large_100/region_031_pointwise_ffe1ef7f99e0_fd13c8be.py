"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: ffe1ef7f99e0
Shape hash: fd13c8be
"""
_shapes_config = "(T([40], f32), T([512, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([40], f32), T([40], f32), T([40], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg61_1: "f32[40]", convolution_13: "f32[512, 40, 28, 28]", arg62_1: "f32[40]", arg63_1: "f32[40]", arg64_1: "f32[40]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_default_1: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_default_1);  convolution_13 = unsqueeze_default_1 = None
        add_tensor: "f32[40]" = torch.ops.aten.add.Tensor(arg62_1, 1e-05);  arg62_1 = None
        sqrt_default: "f32[40]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_default_5: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_default_7: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        return add_tensor_1



def make_inputs():
    return [
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(16056320, dtype=torch.float32, device='cuda').as_strided([512, 40, 28, 28], [31360, 1, 1120, 40]),  # convolution_13
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
