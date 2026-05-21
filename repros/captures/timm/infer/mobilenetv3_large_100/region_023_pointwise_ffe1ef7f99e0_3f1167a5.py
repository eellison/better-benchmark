"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: ffe1ef7f99e0
Shape hash: 3f1167a5
"""
_shapes_config = "(T([80], f32), T([512, 80, 14, 14], f32, stride=(15680, 1, 1120, 80)), T([80], f32), T([80], f32), T([80], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg114_1: "f32[80]", convolution_26: "f32[512, 80, 14, 14]", arg115_1: "f32[80]", arg116_1: "f32[80]", arg117_1: "f32[80]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_default_1);  convolution_26 = unsqueeze_default_1 = None
        add_tensor: "f32[80]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_default: "f32[80]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_default_5: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_default_7: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        return add_tensor_1



def make_inputs():
    return [
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn(8028160, dtype=torch.float32, device='cuda').as_strided([512, 80, 14, 14], [15680, 1, 1120, 80]),  # convolution_26
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
