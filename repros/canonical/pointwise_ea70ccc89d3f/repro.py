"""
Standalone repro captured via capture_hook.
Label: vit_b_16_inference
Pattern hash: ea70ccc89d3f
Shape hash: e06d6fbc
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
    def forward(self, add_84: "f32[1, 197, 768]", getitem_97: "f32[1, 197, 1]", getitem_96: "f32[1, 197, 1]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[1000, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:157 in forward, code: return self.ln(self.layers(self.dropout(input)))
        sub_tensor: "f32[1, 197, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_97);  add_84 = getitem_97 = None
        add_tensor: "f32[1, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_default: "f32[1, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg149_1);  mul_tensor = arg149_1 = None
        add_tensor_1: "f32[1, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg150_1);  mul_tensor_1 = arg150_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:301 in forward, code: x = x[:, 0]
        select_int: "f32[1, 768]" = torch.ops.aten.select.int(add_tensor_1, 1, 0);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:303 in forward, code: x = self.heads(x)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        return (select_int, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
