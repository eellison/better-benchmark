"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 5713f1eca23d
Shape hash: d79e48e7
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f32[4, 8, 1, 1]", getitem_329: "f32[4, 8, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        neg_default: "f32[4, 8, 1, 1]" = torch.ops.aten.neg.default(convolution_2)
        exp_default: "f32[4, 8, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 8, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[4, 8, 1, 1]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_329, mul_tensor);  getitem_329 = None
        sub_tensor: "f32[4, 8, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sub_tensor);  convolution_2 = sub_tensor = None
        add_tensor_1: "f32[4, 8, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[4, 8, 1, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        return mul_tensor_3


def _default_make_inputs():
    return [
    torch.randn([4, 8, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 8, 1, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
