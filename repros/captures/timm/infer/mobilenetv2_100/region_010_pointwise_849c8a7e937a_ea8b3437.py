"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_infer
Pattern hash: 849c8a7e937a
Shape hash: ea8b3437
"""
_shapes_config = "(T([64], f32), T([128, 64, 14, 14], f32, stride=(12544, 1, 896, 64)), T([64], f32), T([64], f32), T([64], f32), T([128, 64, 14, 14], f32, stride=(12544, 1, 896, 64)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg147_1: "f32[64]", convolution_29: "f32[128, 64, 14, 14]", arg148_1: "f32[64]", arg149_1: "f32[64]", arg150_1: "f32[64]", add_58: "f32[128, 64, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_default_1);  convolution_29 = unsqueeze_default_1 = None
        add_tensor: "f32[64]" = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_default: "f32[64]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_default_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 64, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_default_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 64, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[128, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_1, add_58);  add_tensor_1 = add_58 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([128, 64, 14, 14], [12544, 1, 896, 64]),  # convolution_29
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn(1605632, dtype=torch.float32, device='cuda').as_strided([128, 64, 14, 14], [12544, 1, 896, 64]),  # add_58
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
