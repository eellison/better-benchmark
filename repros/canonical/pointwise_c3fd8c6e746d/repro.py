"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_inference
Pattern hash: c3fd8c6e746d
Shape hash: 9c206498
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
    def forward(self, convolution_53: "f32[32, 2304, 7, 7]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        reshape_default: "f32[32, 3, 6, 128, 49]" = torch.ops.aten.reshape.default(convolution_53, _shape_param_0);  convolution_53 = _shape_param_0 = None
        permute_default: "f32[3, 32, 6, 49, 128]" = torch.ops.aten.permute.default(reshape_default, [1, 0, 2, 4, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f32[32, 6, 49, 128]" = unbind_int[0]
        getitem_1: "f32[32, 6, 49, 128]" = unbind_int[1]
        getitem_2: "f32[32, 6, 49, 128]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([32, 2304, 7, 7], dtype=torch.float32, device='cuda'),
    [32, 3, 6, 128, -1],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
