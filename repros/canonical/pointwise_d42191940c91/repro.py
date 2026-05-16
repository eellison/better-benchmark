"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_training
Pattern hash: d42191940c91
Shape hash: d46a8339
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[32, 768]", _shape_param_0, primals_162: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention_pool.py:126 in forward, code: x = x[:, 0]
        full_default: "f32[32, 1, 768]" = torch.ops.aten.full.default([32, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[32, 1, 768]" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 1, 0);  full_default = tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        reshape_default: "f32[32, 768]" = torch.ops.aten.reshape.default(select_scatter_default, _shape_param_0);  select_scatter_default = _shape_param_0 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    [32, 768],  # _shape_param_0
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
