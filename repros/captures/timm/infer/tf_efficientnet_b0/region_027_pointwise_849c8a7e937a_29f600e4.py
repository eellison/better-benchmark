"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 849c8a7e937a
Shape hash: 29f600e4
"""
_shapes_config = "(T([40], f32), T([128, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)), T([40], f32), T([40], f32), T([40], f32), T([128, 40, 28, 28], f32, stride=(31360, 1, 1120, 40)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg92_1: "f32[40]", convolution_24: "f32[128, 40, 28, 28]", arg93_1: "f32[40]", arg94_1: "f32[40]", arg95_1: "f32[40]", add_36: "f32[128, 40, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_default_1: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_default_1);  convolution_24 = unsqueeze_default_1 = None
        add_tensor: "f32[40]" = torch.ops.aten.add.Tensor(arg93_1, 0.001);  arg93_1 = None
        sqrt_default: "f32[40]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[40]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[40]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_default_5: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_default_7: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[128, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_1, add_36);  add_tensor_1 = add_36 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 40, 28, 28], [31360, 1, 1120, 40]),  # convolution_24
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([128, 40, 28, 28], [31360, 1, 1120, 40]),  # add_36
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
