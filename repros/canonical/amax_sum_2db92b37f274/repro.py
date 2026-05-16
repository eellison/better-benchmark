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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[2048, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:233 in forward, code: routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        convert_element_type_default: "f32[2048, 128]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        amax_default: "f32[2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[2048, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[2048, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[2048, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:234 in forward, code: routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_default = torch.ops.aten.topk.default(div_tensor, 8);  div_tensor = None
        getitem: "f32[2048, 8]" = topk_default[0]
        getitem_1: "i64[2048, 8]" = topk_default[1];  topk_default = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
