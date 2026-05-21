"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: a8b0e6ac106c
Shape hash: a73ec0d0
"""
_shapes_config = "(T([48], f32), T([128, 48, 35, 35], f32, stride=(58800, 1, 1680, 48)), T([48], f32), T([48], f32), T([48], f32), T([96], f32), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([96], f32), T([96], f32), T([96], f32), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg102_1: "f32[48]", convolution_20: "f32[128, 48, 35, 35]", arg103_1: "f32[48]", arg104_1: "f32[48]", arg105_1: "f32[48]", arg117_1: "f32[96]", convolution_23: "f32[128, 96, 35, 35]", arg118_1: "f32[96]", arg119_1: "f32[96]", arg120_1: "f32[96]", cat_1: "f32[128, 288, 35, 35]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_default_1: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_default_1);  convolution_20 = unsqueeze_default_1 = None
        add_tensor: "f32[48]" = torch.ops.aten.add.Tensor(arg103_1, 0.001);  arg103_1 = None
        sqrt_default: "f32[48]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[48]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[48]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_default_5: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_default_7: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 48, 35, 35]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_8: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_default_9: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_default_9);  convolution_23 = unsqueeze_default_9 = None
        add_tensor_2: "f32[96]" = torch.ops.aten.add.Tensor(arg118_1, 0.001);  arg118_1 = None
        sqrt_default_1: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_default_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_default_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 96, 35, 35]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_default: "f32[128, 288, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat_1, [3, 3], [1, 1], [1, 1]);  cat_1 = None
        return (relu_default, relu_default_1, avg_pool2d_default)



def make_inputs():
    return [
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn(7526400, dtype=torch.float32, device='cuda').as_strided([128, 48, 35, 35], [58800, 1, 1680, 48]),  # convolution_20
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(15052800, dtype=torch.float32, device='cuda').as_strided([128, 96, 35, 35], [117600, 1, 3360, 96]),  # convolution_23
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(45158400, dtype=torch.float32, device='cuda').as_strided([128, 288, 35, 35], [352800, 1, 10080, 288]),  # cat_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
