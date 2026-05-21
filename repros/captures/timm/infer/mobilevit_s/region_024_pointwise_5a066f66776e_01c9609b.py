"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 5a066f66776e
Shape hash: 01c9609b
"""
_shapes_config = "(T([96], f32), T([128, 96, 32, 32], f32, stride=(98304, 1, 3072, 96)), T([96], f32), T([96], f32), T([96], f32), T([128, 96, 32, 32], f32, stride=(98304, 1, 3072, 96)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg114_1: "f32[96]", convolution_18: "f32[128, 96, 32, 32]", arg115_1: "f32[96]", arg116_1: "f32[96]", arg117_1: "f32[96]", add_44: "f32[128, 96, 32, 32]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_default_1: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_default_1);  convolution_18 = unsqueeze_default_1 = None
        add_tensor: "f32[96]" = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_default: "f32[96]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_default_5: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_default_7: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 96, 32, 32]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 96, 32, 32]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 96, 32, 32]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 96, 32, 32]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        cat_default: "f32[128, 192, 32, 32]" = torch.ops.aten.cat.default([add_44, div_tensor], 1);  add_44 = div_tensor = None
        return cat_default



def make_inputs():
    return [
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([128, 96, 32, 32], [98304, 1, 3072, 96]),  # convolution_18
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(12582912, dtype=torch.float32, device='cuda').as_strided([128, 96, 32, 32], [98304, 1, 3072, 96]),  # add_44
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
