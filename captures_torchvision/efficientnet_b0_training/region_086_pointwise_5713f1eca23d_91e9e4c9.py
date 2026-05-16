"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 5713f1eca23d
Shape hash: 91e9e4c9
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[4, 10, 1, 1]", getitem_269: "f32[4, 10, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        neg_default: "f32[4, 10, 1, 1]" = torch.ops.aten.neg.default(convolution_22)
        exp_default: "f32[4, 10, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[4, 10, 1, 1]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_269, mul_tensor);  getitem_269 = None
        sub_tensor: "f32[4, 10, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sub_tensor);  convolution_22 = sub_tensor = None
        add_tensor_1: "f32[4, 10, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 10, 1, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3



def make_inputs():
    return [
    torch.randn([4, 10, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 10, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
