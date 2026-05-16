"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['2048', '8'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[2048, 8, 2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:129 in get_topk_indices, code: .sum(dim=-1)
        sum_dim_int_list: "f32[2048, 8]" = torch.ops.aten.sum.dim_IntList(getitem, [-1]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:131 in get_topk_indices, code: group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        topk_default = torch.ops.aten.topk.default(sum_dim_int_list, 4, -1, True, False);  sum_dim_int_list = None
        getitem_1: "i64[2048, 4]" = topk_default[1];  topk_default = None
        return getitem_1


def _default_make_inputs():
    return [
    torch.randn([2048, 8, 2], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
