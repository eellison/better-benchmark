"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['2048', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[2048, 8]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:236 in forward, code: routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(getitem, [-1], True)
        div_tensor: "f32[2048, 8]" = torch.ops.aten.div.Tensor(getitem, sum_dim_int_list);  getitem = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:238 in forward, code: routing_weights = routing_weights.to(hidden_states.dtype)
        convert_element_type_default: "bf16[2048, 8]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return convert_element_type_default


def _default_make_inputs():
    return [
    torch.randn([2048, 8], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
