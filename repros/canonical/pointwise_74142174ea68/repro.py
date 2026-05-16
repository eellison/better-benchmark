"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: 74142174ea68
Shape hash: 06e8c943
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_118: "f32[1, 80, 1, 1]", convolution_69: "f32[32, 80, 7, 7]", getitem_119: "f32[1, 80, 1, 1]", primals_380: "f32[80]", primals_381: "f32[80]", getitem_122: "f32[1, 112, 1, 1]", convolution_71: "f32[32, 112, 7, 7]", getitem_123: "f32[1, 112, 1, 1]", primals_392: "f32[112]", primals_393: "f32[112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_default: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_119);  convolution_69 = getitem_119 = None
        mul_tensor: "f32[32, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_380, -1);  primals_380 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_381, -1);  primals_381 = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_2: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
        rsqrt_default_1: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[32, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_123);  convolution_71 = getitem_123 = None
        mul_tensor_2: "f32[32, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_392, -1);  primals_392 = None
        unsqueeze_default_5: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[32, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_393, -1);  primals_393 = None
        unsqueeze_default_7: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[32, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        return (add_tensor_1, add_tensor_3)


def _default_make_inputs():
    return [
    torch.randn([1, 80, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 80, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 80, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([80], dtype=torch.float32, device='cuda'),
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 112, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 112, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
