"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '2304'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_178: "f32[8, 12, 1024, 64]", getitem_180: "f32[8, 12, 1024, 64]", getitem_179: "f32[8, 12, 1024, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:319 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        reshape_default: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default, [8, 1024, 768]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_default_1: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        reshape_default_1: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_1, [8, 1024, 768]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:315 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_default_2: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
        reshape_default_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_default_2, [8, 1024, 768]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:313 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_default: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([reshape_default, reshape_default_2, reshape_default_1], 2);  reshape_default = reshape_default_2 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        reshape_default_3: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_default, [8192, 2304]);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        sum_dim_int_list: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [0], True);  reshape_default_3 = None
        reshape_default_4: "f32[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list, [2304]);  sum_dim_int_list = None
        return reshape_default_4


def _default_make_inputs():
    return [
    torch.randn([8, 12, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 1024, 64], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 1024, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
