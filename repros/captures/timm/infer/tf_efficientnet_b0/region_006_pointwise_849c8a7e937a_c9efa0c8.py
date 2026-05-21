"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 849c8a7e937a
Shape hash: c9efa0c8
"""
_shapes_config = "(T([192], f32), T([128, 192, 7, 7], f32, stride=(9408, 1, 1344, 192)), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 7, 7], f32, stride=(9408, 1, 1344, 192)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg282_1: "f32[192]", convolution_74: "f32[128, 192, 7, 7]", arg283_1: "f32[192]", arg284_1: "f32[192]", arg285_1: "f32[192]", add_133: "f32[128, 192, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_default_1);  convolution_74 = unsqueeze_default_1 = None
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(arg283_1, 0.001);  arg283_1 = None
        sqrt_default: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:338 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, add_133);  add_tensor_1 = add_133 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([128, 192, 7, 7], [9408, 1, 1344, 192]),  # convolution_74
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(1204224, dtype=torch.float32, device='cuda').as_strided([128, 192, 7, 7], [9408, 1, 1344, 192]),  # add_133
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
