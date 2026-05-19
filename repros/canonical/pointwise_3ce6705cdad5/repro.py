"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 3ce6705cdad5
Shape hash: 75daa4cb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_44: "f32[768, 196, 64]", bmm_46: "f32[768, 64, 196]", bmm_47: "f32[768, 196, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        reshape_default: "f32[128, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_44, _shape_param_0);  bmm_44 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        reshape_default_1: "f32[128, 6, 64, 196]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_1);  bmm_46 = _shape_param_1 = None
        reshape_default_2: "f32[128, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_47, _shape_param_2);  bmm_47 = _shape_param_2 = None
        permute_default: "f32[128, 6, 196, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_default: "f32[384, 6, 196, 64]" = torch.ops.aten.cat.default([reshape_default_2, permute_default, reshape_default]);  reshape_default_2 = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 128, 6, 196, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_default_1: "f32[128, 3, 6, 64, 196]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0, 2, 4, 3]);  reshape_default_3 = None
        clone_default: "f32[128, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[128, 1152, 14, 14]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        return reshape_default_4


def _default_make_inputs():
    return [
    torch.randn([768, 196, 64], dtype=torch.float32, device='cuda'),
    torch.randn([768, 64, 196], dtype=torch.float32, device='cuda'),
    torch.randn([768, 196, 64], dtype=torch.float32, device='cuda'),
    [128, 6, 196, 64],  # _shape_param_0
    [128, 6, 64, 196],  # _shape_param_1
    [128, 6, 196, 64],  # _shape_param_2
    [3, 128, 6, 196, 64],  # _shape_param_3
    [128, 1152, 14, 14],  # _shape_param_4
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
