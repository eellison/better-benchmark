"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 849c8a7e937a
Shape hash: 5e1dfb6f
"""
_shapes_config = "(T([16], f32), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)), T([16], f32), T([16], f32), T([16], f32), T([512, 16, 112, 112], f32, stride=(200704, 1, 1792, 16)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg12_1: "f32[16]", convolution_2: "f32[512, 16, 112, 112]", arg13_1: "f32[16]", arg14_1: "f32[16]", arg15_1: "f32[16]", div: "f32[512, 16, 112, 112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_default_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_default_1);  convolution_2 = unsqueeze_default_1 = None
        add_tensor: "f32[16]" = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_default: "f32[16]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[16]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[16]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_default_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_default_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:230 in forward, code: x = self.drop_path(x) + shortcut
        add_tensor_2: "f32[512, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_tensor_1, div);  add_tensor_1 = div = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # convolution_2
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([512, 16, 112, 112], [200704, 1, 1792, 16]),  # div
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
