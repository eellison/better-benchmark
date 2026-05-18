"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_137: "f32[192, 128, 128]", gt_6: "b8[32, 6, 128, 128]", div_3: "f32[32, 6, 128, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:443 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        reshape_default: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [32, 6, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:437 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:436 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_tensor_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_3);  mul_tensor_1 = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_default: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:433 in forward, code: scores += position_bias_masked
        reshape_default_1: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_default, [192, 128, 128]);  fma_default = None
        reshape_default_2: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_1, [32, 6, 128, 128]);  reshape_default_1 = None
        reshape_default_3: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(reshape_default_2, [192, 128, 128]);  reshape_default_2 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [32, 6, 128, 128], dtype=torch.bool, device='cuda'),
    torch.randn([32, 6, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
