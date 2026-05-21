"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_infer
Pattern hash: 42ab411e21f6
Shape hash: 1ae9c98a
"""
_shapes_config = "(T([96], f32), T([128, 96, 56, 56], f32, stride=(301056, 1, 5376, 96)), T([96], f32), T([96], f32), T([96], f32), T([96], f32), T([128, 96, 56, 56], f32, stride=(301056, 1, 5376, 96)), T([96], f32), T([96], f32), T([96], f32), T([96], f32), T([128, 96, 56, 56], f32, stride=(301056, 1, 5376, 96)), T([96], f32), T([96], f32), T([96], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg26_1: "f32[96]", convolution_4: "f32[128, 96, 56, 56]", arg27_1: "f32[96]", arg28_1: "f32[96]", arg29_1: "f32[96]", arg31_1: "f32[96]", convolution_5: "f32[128, 96, 56, 56]", arg32_1: "f32[96]", arg33_1: "f32[96]", arg34_1: "f32[96]", arg21_1: "f32[96]", relu_1: "f32[128, 96, 56, 56]", arg22_1: "f32[96]", arg23_1: "f32[96]", arg24_1: "f32[96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg26_1, -1);  arg26_1 = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_default_1);  convolution_4 = unsqueeze_default_1 = None
        add_tensor: "f32[96]" = torch.ops.aten.add.Tensor(arg27_1, 1e-05);  arg27_1 = None
        sqrt_default: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_default_5: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_default_7: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        unsqueeze_default_8: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_default_9: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_default_9);  convolution_5 = unsqueeze_default_9 = None
        add_tensor_2: "f32[96]" = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_default_1: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_default_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_default_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_tensor_4: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor_3);  add_tensor_1 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, -1);  arg21_1 = None
        unsqueeze_default_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        sub_tensor_2: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(relu_1, unsqueeze_default_17);  relu_1 = unsqueeze_default_17 = None
        add_tensor_5: "f32[96]" = torch.ops.aten.add.Tensor(arg22_1, 1e-05);  arg22_1 = None
        sqrt_default_2: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor_5);  add_tensor_5 = None
        reciprocal_default_2: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        mul_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_default_21: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_default_23: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_6: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_tensor_7: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_4, add_tensor_6);  add_tensor_4 = add_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_default: "f32[128, 96, 56, 56]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # convolution_4
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # convolution_5
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(38535168, dtype=torch.float32, device='cuda').as_strided([128, 96, 56, 56], [301056, 1, 5376, 96]),  # relu_1
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
