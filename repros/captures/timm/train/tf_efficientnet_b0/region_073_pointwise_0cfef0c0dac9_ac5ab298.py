"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 0cfef0c0dac9
Shape hash: ac5ab298
"""
_shapes_config = "(T([128, 10, 1, 1], f32), T([128, 10, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[128, 10, 1, 1]", getitem_269: "f32[128, 10, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_default: "f32[128, 10, 1, 1]" = torch.ops.aten.neg.default(convolution_22)
        exp_default: "f32[128, 10, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 10, 1, 1]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_269, mul_tensor);  getitem_269 = None
        sub_tensor: "f32[128, 10, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sub_tensor);  convolution_22 = sub_tensor = None
        add_tensor_1: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3



def make_inputs():
    return [
    torch.randn([128, 10, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 10, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
