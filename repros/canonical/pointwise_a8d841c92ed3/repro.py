"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_training
Pattern hash: a8d841c92ed3
Shape hash: 645a93d0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_178: "f32[32, 12, 197, 64]", getitem_179: "f32[32, 12, 197, 64]", getitem_180: "f32[32, 12, 197, 64]", _shape_param_0, _shape_param_1, _shape_param_2, primals_11: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:220 in forward, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        cat_default: "f32[96, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        reshape_default: "f32[3, 32, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:219 in forward, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default: "f32[32, 197, 3, 12, 64]" = torch.ops.aten.permute.default(reshape_default, [1, 3, 0, 2, 4]);  reshape_default = None
        clone_default: "f32[32, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 197, 2304]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/beit.py:218 in forward, code: qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        reshape_default_2: "f32[6304, 2304]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_2: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        return (reshape_default_2, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_178
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_179
    torch.randn(4841472, dtype=torch.float32, device='cuda').as_strided([32, 12, 197, 64], [151296, 64, 768, 1]),  # getitem_180
    [3, 32, 12, 197, 64],  # _shape_param_0
    [32, 197, 2304],  # _shape_param_1
    [6304, 2304],  # _shape_param_2
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
