"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_56: "i64[16384]", getitem_57: "i64[16384]", view_78: "bf16[2048, 2048]", arg45_1: "bf16[128, 1536, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:416 in grouped_mm_experts_forward, code: sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
        ge_scalar: "b8[16384]" = torch.ops.aten.ge.Scalar(getitem_56, 128)
        unsqueeze_default: "b8[16384, 1]" = torch.ops.aten.unsqueeze.default(ge_scalar, -1);  ge_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        div_tensor_mode: "i64[16384]" = torch.ops.aten.div.Tensor_mode(getitem_57, 8, rounding_mode = 'floor');  getitem_57 = None
        index_tensor: "bf16[16384, 2048]" = torch.ops.aten.index.Tensor(view_78, [div_tensor_mode]);  view_78 = div_tensor_mode = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_default, full_default, index_tensor);  unsqueeze_default = full_default = index_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:364 in _grouped_linear, code: out = _grouped_mm(input, weight.transpose(-2, -1), offs=offs)
        permute_default: "bf16[128, 2048, 1536]" = torch.ops.aten.permute.default(arg45_1, [0, 2, 1]);  arg45_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:397 in grouped_mm_experts_forward, code: histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
        convert_element_type_default: "i32[16384]" = torch.ops.prims.convert_element_type.default(getitem_56, torch.int32);  getitem_56 = None
        return (where_self, permute_default, convert_element_type_default)


def _default_make_inputs():
    return [
    torch.randint(0, 100, [16384], dtype=torch.int64, device='cuda'),
    torch.randint(0, 100, [16384], dtype=torch.int64, device='cuda'),
    torch.randn([2048, 2048], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 1536, 2048], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
