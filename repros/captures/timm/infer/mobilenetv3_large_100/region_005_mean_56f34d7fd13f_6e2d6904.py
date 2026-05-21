"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: 56f34d7fd13f
Shape hash: 6e2d6904
"""
_shapes_config = "(T([960], f32), T([512, 960, 7, 7], f32, stride=(47040, 1, 6720, 960)), T([960], f32), T([960], f32), T([960], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg245_1: "f32[960]", convolution_57: "f32[512, 960, 7, 7]", arg246_1: "f32[960]", arg247_1: "f32[960]", arg248_1: "f32[960]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_default_1);  convolution_57 = unsqueeze_default_1 = None
        add_tensor: "f32[960]" = torch.ops.aten.add.Tensor(arg246_1, 1e-05);  arg246_1 = None
        sqrt_default: "f32[960]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_default_5: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, -1);  arg248_1 = None
        unsqueeze_default_7: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        add_tensor_2: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, 3)
        clamp_min_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default: "f32[512, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_3: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_1, clamp_max_default);  add_tensor_1 = clamp_max_default = None
        div_tensor: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_3, 6);  mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[512, 960, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [2, 3], True);  div_tensor = None
        return mean_dim



def make_inputs():
    return [
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn(24084480, dtype=torch.float32, device='cuda').as_strided([512, 960, 7, 7], [47040, 1, 6720, 960]),  # convolution_57
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
