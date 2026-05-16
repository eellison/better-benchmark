"""
Standalone repro captured via capture_hook.
Label: timm_convnext_tiny
Pattern hash: 2de2263121b9
Shape hash: ee975090
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, permute_66: "f32[8, 14, 14, 384]", getitem_37: "f32[8, 14, 14, 1]", getitem_36: "f32[8, 14, 14, 1]", arg148_1: "f32[384]", arg149_1: "f32[384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[8, 14, 14, 384]" = torch.ops.aten.sub.Tensor(permute_66, getitem_37);  permute_66 = getitem_37 = None
        add_tensor: "f32[8, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
        rsqrt_default: "f32[8, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 14, 14, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg148_1);  mul_tensor = arg148_1 = None
        add_tensor_1: "f32[8, 14, 14, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg149_1);  mul_tensor_1 = arg149_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[8, 384, 14, 14]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([8, 14, 14, 384], dtype=torch.float32, device='cuda'),
    torch.randn([8, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 14, 14, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
