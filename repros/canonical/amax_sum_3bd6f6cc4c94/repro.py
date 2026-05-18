"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=amax, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[2048, 4]", getitem_1: "i64[2048, 4]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        full_default: "f32[2048, 32]" = torch.ops.aten.full.default([2048, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:156 in forward, code: router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(getitem, [1], True)
        sub_tensor: "f32[2048, 4]" = torch.ops.aten.sub.Tensor(getitem, amax_default);  getitem = amax_default = None
        exp_default: "f32[2048, 4]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[2048, 4]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:157 in forward, code: router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        scatter_src: "f32[2048, 32]" = torch.ops.aten.scatter.src(full_default, 1, getitem_1, div_tensor);  full_default = getitem_1 = div_tensor = None
        return scatter_src


def _default_make_inputs():
    return [
    torch.randn([2048, 4], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2048, 4], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
