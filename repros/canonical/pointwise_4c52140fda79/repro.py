"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 4c52140fda79
Shape hash: b22ff33f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_94: "f32[1, 320, 1, 1]", convolution_79: "f32[4, 320, 7, 7]", getitem_95: "f32[1, 320, 1, 1]", primals_352: "f32[320]", primals_353: "f32[320]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/efficientnet.py:165 in forward, code: result = self.block(input)
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_95);  convolution_79 = getitem_95 = None
        mul_tensor: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_352, -1);  primals_352 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 320, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1);  primals_353 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 320, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        return add_tensor_1


def _default_make_inputs():
    return [
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 320, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
