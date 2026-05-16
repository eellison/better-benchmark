"""
Standalone repro captured via capture_hook.
Label: timm_convnext_tiny
Pattern hash: 1ecc9387d6bd
Shape hash: d07a0b32
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, permute_80: "f32[8, 1, 1, 768]", getitem_45: "f32[8, 1, 1, 1]", getitem_44: "f32[8, 1, 1, 1]", arg179_1: "f32[768]", arg180_1: "f32[768]", _shape_param_0, arg181_1: "f32[1000, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_tensor: "f32[8, 1, 1, 768]" = torch.ops.aten.sub.Tensor(permute_80, getitem_45);  permute_80 = getitem_45 = None
        add_tensor: "f32[8, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
        rsqrt_default: "f32[8, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1, 1, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg179_1);  mul_tensor = arg179_1 = None
        add_tensor_1: "f32[8, 1, 1, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg180_1);  mul_tensor_1 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_default: "f32[8, 768, 1, 1]" = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        reshape_default: "f32[8, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_default_1: "f32[768, 1000]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    [8, 768],  # _shape_param_0
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
