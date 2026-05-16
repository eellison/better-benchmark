"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_72: "f32[256, 12, 128, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:382 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        permute_default: "f32[256, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(permute_default, [256, 128, 768]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:388 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(reshape_default, [32768, 768]);  reshape_default = None
        permute_default_1: "f32[768, 32768]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0], True);  reshape_default_1 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, [768]);  sum_dim_int_list = None
        return (permute_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([256, 12, 128, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
