"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_infer
Pattern hash: 226a109e3e14
Shape hash: fcaa3727
"""
_shapes_config = "(T([160], f32), T([128, 160, 8, 8], f32, stride=(10240, 1, 1280, 160)), T([160], f32), T([160], f32), T([160], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg269_1: "f32[160]", convolution_33: "f32[128, 160, 8, 8]", arg270_1: "f32[160]", arg271_1: "f32[160]", arg272_1: "f32[160]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_default_1);  convolution_33 = unsqueeze_default_1 = None
        add_tensor: "f32[160]" = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_default: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_default_5: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_default_7: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_default: "f32[128, 160, 8, 8]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[128, 160, 8, 8]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[128, 160, 8, 8]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 160, 8, 8]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn(1310720, dtype=torch.float32, device='cuda').as_strided([128, 160, 8, 8], [10240, 1, 1280, 160]),  # convolution_33
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
