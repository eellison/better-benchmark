"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: aa194b04264a
Shape hash: 641a711e
"""
_shapes_config = "(T([144], f32), T([128, 144, 56, 56], f32, stride=(451584, 1, 8064, 144)), T([144], f32), T([144], f32), T([144], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg59_1: "f32[144]", convolution_15: "f32[128, 144, 56, 56]", arg60_1: "f32[144]", arg61_1: "f32[144]", arg62_1: "f32[144]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_default_1: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_default_1);  convolution_15 = unsqueeze_default_1 = None
        add_tensor: "f32[144]" = torch.ops.aten.add.Tensor(arg60_1, 0.001);  arg60_1 = None
        sqrt_default: "f32[144]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[144]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[144]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_default_5: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_default_7: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 144, 56, 56]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 144, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 144, 56, 56]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 144, 59, 59]" = torch.ops.aten.constant_pad_nd.default(div_tensor, [1, 2, 1, 2], 0.0);  div_tensor = None
        return constant_pad_nd_default



def make_inputs():
    return [
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn(57802752, dtype=torch.float32, device='cuda').as_strided([128, 144, 56, 56], [451584, 1, 8064, 144]),  # convolution_15
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    torch.randn([144], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
