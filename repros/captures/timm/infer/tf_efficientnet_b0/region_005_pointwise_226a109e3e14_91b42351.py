"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 226a109e3e14
Shape hash: 91b42351
"""
_shapes_config = "(T([1152], f32), T([128, 1152, 7, 7], f32, stride=(56448, 1, 8064, 1152)), T([1152], f32), T([1152], f32), T([1152], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg287_1: "f32[1152]", convolution_75: "f32[128, 1152, 7, 7]", arg288_1: "f32[1152]", arg289_1: "f32[1152]", arg290_1: "f32[1152]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_1: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_default_1);  convolution_75 = unsqueeze_default_1 = None
        add_tensor: "f32[1152]" = torch.ops.aten.add.Tensor(arg288_1, 0.001);  arg288_1 = None
        sqrt_default: "f32[1152]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1152]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1152]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_default_5: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_default_7: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn(7225344, dtype=torch.float32, device='cuda').as_strided([128, 1152, 7, 7], [56448, 1, 8064, 1152]),  # convolution_75
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
