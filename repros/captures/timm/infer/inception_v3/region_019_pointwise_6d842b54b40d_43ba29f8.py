"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: 6d842b54b40d
Shape hash: 43ba29f8
"""
_shapes_config = "(T([64], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([64], f32), T([64], f32), T([64], f32), T([96], f32), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([96], f32), T([96], f32), T([96], f32), T([64], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([64], f32), T([64], f32), T([64], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg97_1: "f32[64]", convolution_19: "f32[128, 64, 35, 35]", arg98_1: "f32[64]", arg99_1: "f32[64]", arg100_1: "f32[64]", arg107_1: "f32[64]", convolution_21: "f32[128, 64, 35, 35]", arg108_1: "f32[64]", arg109_1: "f32[64]", arg110_1: "f32[64]", arg122_1: "f32[96]", convolution_24: "f32[128, 96, 35, 35]", arg123_1: "f32[96]", arg124_1: "f32[96]", arg125_1: "f32[96]", arg127_1: "f32[64]", convolution_25: "f32[128, 64, 35, 35]", arg128_1: "f32[64]", arg129_1: "f32[64]", arg130_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_default_1);  convolution_19 = unsqueeze_default_1 = None
        add_tensor: "f32[64]" = torch.ops.aten.add.Tensor(arg98_1, 0.001);  arg98_1 = None
        sqrt_default: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_default_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_default_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_default_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_default_9);  convolution_21 = unsqueeze_default_9 = None
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(arg108_1, 0.001);  arg108_1 = None
        sqrt_default_1: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_default_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_default_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_default_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        sub_tensor_2: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_default_17);  convolution_24 = unsqueeze_default_17 = None
        add_tensor_4: "f32[96]" = torch.ops.aten.add.Tensor(arg123_1, 0.001);  arg123_1 = None
        sqrt_default_2: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor_4);  add_tensor_4 = None
        reciprocal_default_2: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        mul_tensor_7: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_default_21: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_default_23: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_5: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 96, 35, 35]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_default_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        sub_tensor_3: "f32[128, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_default_25);  convolution_25 = unsqueeze_default_25 = None
        add_tensor_6: "f32[64]" = torch.ops.aten.add.Tensor(arg128_1, 0.001);  arg128_1 = None
        sqrt_default_3: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor_6);  add_tensor_6 = None
        reciprocal_default_3: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, -1);  mul_tensor_9 = None
        unsqueeze_default_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        mul_tensor_10: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_27);  sub_tensor_3 = unsqueeze_default_27 = None
        unsqueeze_default_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_default_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_11: "f32[128, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_29);  mul_tensor_10 = unsqueeze_default_29 = None
        unsqueeze_default_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_default_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_7: "f32[128, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_31);  mul_tensor_11 = unsqueeze_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 64, 35, 35]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        cat_default: "f32[128, 288, 35, 35]" = torch.ops.aten.cat.default([relu_default, relu_default_1, relu_default_2, relu_default_3], 1);  relu_default = relu_default_1 = relu_default_2 = relu_default_3 = None
        return cat_default



def make_inputs():
    return [
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_19
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_21
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(15052800, dtype=torch.float32, device='cuda').as_strided([128, 96, 35, 35], [117600, 1, 3360, 96]),  # convolution_24
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(10035200, dtype=torch.float32, device='cuda').as_strided([128, 64, 35, 35], [78400, 1, 2240, 64]),  # convolution_25
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
