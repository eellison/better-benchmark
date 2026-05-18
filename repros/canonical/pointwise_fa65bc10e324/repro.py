"""
Standalone repro captured via capture_hook.
Label: vit_b_16_training
Pattern hash: fa65bc10e324
Shape hash: d79c10eb
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
    def forward(self, getitem_88: "f32[4, 197, 1]", clone_88: "f32[4, 197, 768]", getitem_89: "f32[4, 197, 1]", primals_138: "f32[768]", primals_139: "f32[768]", primals_141: "f32[2304, 768]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:112 in forward, code: x = self.ln_1(input)
        add_tensor: "f32[4, 197, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_default: "f32[4, 197, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 197, 768]" = torch.ops.aten.sub.Tensor(clone_88, getitem_89);  clone_88 = getitem_89 = None
        mul_tensor: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_138);  mul_tensor = primals_138 = None
        add_tensor_1: "f32[4, 197, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_139);  mul_tensor_1 = primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vision_transformer.py:113 in forward, code: x, _ = self.self_attention(x, x, x, need_weights=False)
        permute_default: "f32[197, 4, 768]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        clone_default: "f32[197, 4, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default: "f32[788, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        return (permute_default_1, reshape_default)


def _default_make_inputs():
    return [
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 197, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    [788, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
