"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 5713f1eca23d
Shape hash: edde8496
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_7: "f32[4, 4, 1, 1]", getitem_314: "f32[4, 4, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        neg_default: "f32[4, 4, 1, 1]" = torch.ops.aten.neg.default(convolution_7)
        exp_default: "f32[4, 4, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 4, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[4, 4, 1, 1]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_314, mul_tensor);  getitem_314 = None
        sub_tensor: "f32[4, 4, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sub_tensor);  convolution_7 = sub_tensor = None
        add_tensor_1: "f32[4, 4, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 4, 1, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3



def make_inputs():
    return [
    torch.randn([4, 4, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 4, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
