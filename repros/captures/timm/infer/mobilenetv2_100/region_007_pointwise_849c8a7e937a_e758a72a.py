"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_infer
Pattern hash: 849c8a7e937a
Shape hash: e758a72a
"""
_shapes_config = "(T([96], f32), T([128, 96, 14, 14], f32, stride=(18816, 1, 1344, 96)), T([96], f32), T([96], f32), T([96], f32), T([128, 96, 14, 14], f32, stride=(18816, 1, 1344, 96)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg192_1: "f32[96]", convolution_38: "f32[128, 96, 14, 14]", arg193_1: "f32[96]", arg194_1: "f32[96]", arg195_1: "f32[96]", add_78: "f32[128, 96, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_default_1);  convolution_38 = unsqueeze_default_1 = None
        add_tensor: "f32[96]" = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_default: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_default_5: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 96, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_default_7: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 96, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[128, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, add_78);  add_tensor_1 = add_78 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([128, 96, 14, 14], [18816, 1, 1344, 96]),  # convolution_38
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(2408448, dtype=torch.float32, device='cuda').as_strided([128, 96, 14, 14], [18816, 1, 1344, 96]),  # add_78
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
