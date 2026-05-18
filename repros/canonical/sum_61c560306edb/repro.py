"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '30000'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 512, 30000]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:859 in forward, code: hidden_states = self.decoder(hidden_states)
        reshape_default: "f32[4096, 30000]" = torch.ops.aten.reshape.default(tangents_1, _shape_param_0);  tangents_1 = _shape_param_0 = None
        permute_default: "f32[30000, 4096]" = torch.ops.aten.permute.default(reshape_default, [1, 0])
        sum_dim_int_list: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0], True);  reshape_default = None
        reshape_default_1: "f32[30000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 512, 30000], dtype=torch.float32, device='cuda'),
    [4096, 30000],  # _shape_param_0
    [30000],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
