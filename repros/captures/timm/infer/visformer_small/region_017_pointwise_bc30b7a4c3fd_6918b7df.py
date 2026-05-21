"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: bc30b7a4c3fd
Shape hash: 6918b7df
"""
_shapes_config = "(T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32), T([192], f32), T([1, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg8_1: "f32[192]", convolution_1: "f32[128, 192, 28, 28]", arg9_1: "f32[192]", arg10_1: "f32[192]", arg11_1: "f32[192]", arg12_1: "f32[1, 192, 28, 28]", arg13_1: "f32[192]", arg14_1: "f32[192]", arg15_1: "f32[192]", arg16_1: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_default_1);  convolution_1 = unsqueeze_default_1 = None
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_default: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        add_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, arg12_1);  add_tensor_1 = arg12_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_2, unsqueeze_default_9);  add_tensor_2 = unsqueeze_default_9 = None
        add_tensor_3: "f32[192]" = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_default_1: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_3);  add_tensor_3 = None
        reciprocal_default_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_default_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_default_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_4: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        return add_tensor_4



def make_inputs():
    return [
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # convolution_1
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(150528, dtype=torch.float32, device='cuda').as_strided([1, 192, 28, 28], [150528, 1, 5376, 192]),  # arg12_1
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
