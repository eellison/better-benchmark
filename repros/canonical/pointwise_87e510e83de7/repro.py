"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: 87e510e83de7
Shape hash: 29106684
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_144: "f32[1, 480, 1, 1]", convolution_84: "f32[32, 480, 7, 7]", getitem_145: "f32[1, 480, 1, 1]", primals_462: "f32[480]", primals_463: "f32[480]", relu_35: "f32[32, 480, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_default: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_145);  convolution_84 = getitem_145 = None
        mul_tensor: "f32[32, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1);  primals_462 = None
        unsqueeze_default_1: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_default_3: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[32, 480, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[32, 960, 7, 7]" = torch.ops.aten.cat.default([relu_35, relu_default], 1);  relu_35 = relu_default = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 480, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([32, 480, 7, 7], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
