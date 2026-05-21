"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: 92179feb9407
Shape hash: 15fbcc21
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([384], f32), T([384], f32), T([384], f32), T([384], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_63: "f32[128, 384, 14, 14]", convolution_37: "f32[128, 384, 14, 14]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_63, convolution_37);  add_63 = convolution_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_tensor, unsqueeze_default_1);  add_tensor = unsqueeze_default_1 = None
        add_tensor_1: "f32[384]" = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_default: "f32[384]" = torch.ops.aten.sqrt.default(add_tensor_1);  add_tensor_1 = None
        reciprocal_default: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # add_63
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_37
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
