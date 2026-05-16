"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_training
Pattern hash: 416d9619632f
Shape hash: cc253168
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_339: "f32[32, 128, 56, 56]", getitem_342: "f32[32, 128, 56, 56]", convolution_3: "f32[32, 128, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        add_tensor: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_tensor: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0);  add_tensor = None
        neg_default: "f32[32, 128, 56, 56]" = torch.ops.aten.neg.default(convolution_3)
        exp_default: "f32[32, 128, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[32, 128, 56, 56]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = None
        sub_tensor: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(convolution_3, sub_tensor);  convolution_3 = sub_tensor = None
        add_tensor_2: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        return mul_tensor_4


def _default_make_inputs():
    return [
    torch.randn([32, 128, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([32, 128, 56, 56], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
