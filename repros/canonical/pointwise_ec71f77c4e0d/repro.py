"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_280: "f32[8192, 2048]", gt_4: "b8[8, 1024, 2048]", le_11: "b8[8, 1024, 2048]", full_default_2: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:297 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_280, [8, 1024, 2048]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:290 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_tensor: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:289 in forward, code: hidden_states = self.act(hidden_states)
        where_self: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_11, full_default_2, mul_tensor_1);  le_11 = full_default_2 = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:288 in forward, code: hidden_states = self.wi(hidden_states)
        reshape_default_1: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_self, [8192, 2048]);  where_self = None
        permute_default: "f32[2048, 8192]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0]);  reshape_default_1 = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randn([8192, 2048], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randint(0, 2, [8, 1024, 2048], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
