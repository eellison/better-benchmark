"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, gt_1: "b8[64, 128, 1024]", tangents_1: "f32[64, 128, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:395 in forward, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[64, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:394 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_tensor_1, [8192, 1024]);  mul_tensor_1 = None
        permute_default: "f32[1024, 8192]" = torch.ops.aten.permute.default(reshape_default, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(reshape_default, [0], True);  reshape_default = None
        reshape_default_1: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1024]);  sum_dim_int_list = None
        return (permute_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [64, 128, 1024], dtype=torch.bool, device='cuda'),
    torch.randn([64, 128, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
