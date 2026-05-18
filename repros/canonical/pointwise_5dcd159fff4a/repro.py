"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_3: "i64[2048, 4]", add: "f32[2048, 256]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:132 in get_topk_indices, code: group_mask = torch.zeros_like(group_scores)
        full_default: "f32[2048, 8]" = torch.ops.aten.full.default([2048, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:133 in get_topk_indices, code: group_mask.scatter_(1, group_idx, 1)
        scatter_value: "f32[2048, 8]" = torch.ops.aten.scatter.value(full_default, 1, getitem_3, 1);  full_default = getitem_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:136 in get_topk_indices, code: .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
        unsqueeze_default: "f32[2048, 8, 1]" = torch.ops.aten.unsqueeze.default(scatter_value, -1);  scatter_value = None
        expand_default: "f32[2048, 8, 32]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:137 in get_topk_indices, code: .reshape(-1, self.n_routed_experts)
        clone_default: "f32[2048, 8, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default: "f32[2048, 256]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:139 in get_topk_indices, code: scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        convert_element_type_default: "b8[2048, 256]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.bool);  reshape_default = None
        bitwise_not_default: "b8[2048, 256]" = torch.ops.aten.bitwise_not.default(convert_element_type_default);  convert_element_type_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[2048, 256]" = torch.ops.aten.where.self(bitwise_not_default, full_default_1, add);  bitwise_not_default = full_default_1 = add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deepseek_v3/modeling_deepseek_v3.py:140 in get_topk_indices, code: topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_default = torch.ops.aten.topk.default(where_self, 8, -1, True, False);  where_self = None
        getitem: "i64[2048, 8]" = topk_default[1];  topk_default = None
        return getitem


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2048, 4], dtype=torch.int64, device='cuda'),
    torch.randn([2048, 256], dtype=torch.float32, device='cuda'),
    [-1, 8, 32],  # _shape_param_0
    [2048, 256],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
