"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: 3698b0e606e1
Shape hash: 044e8c0f
"""
_shapes_config = "(T([448], f32), T([128, 448, 8, 8], f32, stride=(28672, 1, 3584, 448)), T([448], f32), T([448], f32), T([448], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg447_1: "f32[448]", convolution_89: "f32[128, 448, 8, 8]", arg448_1: "f32[448]", arg449_1: "f32[448]", arg450_1: "f32[448]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg447_1, -1);  arg447_1 = None
        unsqueeze_default_1: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_default_1);  convolution_89 = unsqueeze_default_1 = None
        add_tensor: "f32[448]" = torch.ops.aten.add.Tensor(arg448_1, 0.001);  arg448_1 = None
        sqrt_default: "f32[448]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_default_5: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 448, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_default_7: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 448, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 448, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn(3670016, dtype=torch.float32, device='cuda').as_strided([128, 448, 8, 8], [28672, 1, 3584, 448]),  # convolution_89
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
