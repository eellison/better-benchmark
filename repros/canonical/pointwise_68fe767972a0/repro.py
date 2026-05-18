"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_training
Pattern hash: 68fe767972a0
Shape hash: 17aab4d7
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
    def forward(self, getitem_142: "f32[32, 12, 1, 64]", _shape_param_0, _shape_param_1, primals_152: "f32[768, 768]", getitem_190: "f32[32, 12, 256, 64]", getitem_191: "f32[32, 12, 256, 64]", getitem_192: "f32[32, 12, 256, 64]", _shape_param_2, _shape_param_3, _shape_param_4, primals_7: "f32[2304, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:102 in forward, code: q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
        permute_default: "f32[32, 1, 12, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        reshape_default: "f32[32, 1, 768]" = torch.ops.aten.reshape.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        reshape_default_1: "f32[32, 768]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[96, 12, 256, 64]" = torch.ops.aten.cat.default([getitem_190, getitem_191, getitem_192]);  getitem_190 = getitem_191 = getitem_192 = None
        reshape_default_2: "f32[3, 32, 12, 256, 64]" = torch.ops.aten.reshape.default(cat_default, _shape_param_2);  cat_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_default_3: "f32[32, 256, 3, 12, 64]" = torch.ops.aten.permute.default(reshape_default_2, [1, 3, 0, 2, 4]);  reshape_default_2 = None
        clone_default: "f32[32, 256, 3, 12, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        reshape_default_3: "f32[32, 256, 2304]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        reshape_default_4: "f32[8192, 2304]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_4: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_5: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_default_4, [1, 0]);  permute_default_4 = None
        return (reshape_default_1, permute_default_2, reshape_default_4, permute_default_5)


def _default_make_inputs():
    return [
    torch.randn([32, 12, 1, 64], dtype=torch.float32, device='cuda'),
    [32, 1, 768],  # _shape_param_0
    [32, 768],  # _shape_param_1
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 12, 256, 64], [196608, 64, 768, 1]),  # getitem_190
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 12, 256, 64], [196608, 64, 768, 1]),  # getitem_191
    torch.randn(6291456, dtype=torch.float32, device='cuda').as_strided([32, 12, 256, 64], [196608, 64, 768, 1]),  # getitem_192
    [3, 32, 12, 256, 64],  # _shape_param_2
    [32, 256, 2304],  # _shape_param_3
    [8192, 2304],  # _shape_param_4
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
