"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_infer
Pattern hash: e0eff9609cf0
Shape hash: b05c9ba6
"""
_shapes_config = "(T([576], f32), T([128, 576, 14, 14], f32, stride=(112896, 1, 8064, 576)), T([576], f32), T([576], f32), T([576], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg197_1: "f32[576]", convolution_39: "f32[128, 576, 14, 14]", arg198_1: "f32[576]", arg199_1: "f32[576]", arg200_1: "f32[576]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_default_1: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_default_1);  convolution_39 = unsqueeze_default_1 = None
        add_tensor: "f32[576]" = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_default: "f32[576]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_default_5: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_default_7: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        clamp_min_default: "f32[128, 576, 14, 14]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0.0);  add_tensor_1 = None
        clamp_max_default: "f32[128, 576, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        return clamp_max_default



def make_inputs():
    return [
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn(14450688, dtype=torch.float32, device='cuda').as_strided([128, 576, 14, 14], [112896, 1, 8064, 576]),  # convolution_39
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
