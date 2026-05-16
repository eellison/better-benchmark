"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 041697ce161e
Shape hash: e86c9cef
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_90: "f32[1, 1152, 1, 1]", convolution_75: "f32[4, 1152, 7, 7]", getitem_91: "f32[1, 1152, 1, 1]", primals_336: "f32[1152]", primals_337: "f32[1152]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        add_tensor: "f32[1, 1152, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_default: "f32[1, 1152, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91);  convolution_75 = getitem_91 = None
        mul_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1);  primals_336 = None
        unsqueeze_default_1: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_default_3: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        neg_default: "f32[4, 1152, 7, 7]" = torch.ops.aten.neg.default(add_tensor_1)
        exp_default: "f32[4, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[4, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([1, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1152, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1152, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    torch.randn([1152], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
