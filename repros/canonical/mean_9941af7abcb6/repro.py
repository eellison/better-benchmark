"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: 9941af7abcb6
Shape hash: 83edad2e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_116: "f32[1, 672, 1, 1]", convolution_66: "f32[32, 672, 7, 7]", getitem_117: "f32[1, 672, 1, 1]", primals_370: "f32[672]", primals_371: "f32[672]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        add_tensor: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_default: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_117);  convolution_66 = getitem_117 = None
        mul_tensor: "f32[32, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_370, -1);  primals_370 = None
        unsqueeze_default_1: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[32, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_371, -1);  primals_371 = None
        unsqueeze_default_3: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[32, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_dim: "f32[32, 672, 1, 1]" = torch.ops.aten.mean.dim(add_tensor_1, [2, 3], True);  add_tensor_1 = None
        return mean_dim


def _default_make_inputs():
    return [
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 672, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
