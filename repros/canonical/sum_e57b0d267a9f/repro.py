"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['8', '24', '512', '1'], reduction_ranges=[]
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
    def forward(self, bmm_49: "f32[192, 512, 512]", gt_70: "b8[8, 24, 512, 512]", div_47: "f32[8, 24, 512, 512]", full_default_74: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        reshape_default: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [8, 24, 512, 512]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:265 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_default: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_tensor_2: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_47);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_default: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_default_74, fma_default);  full_default = full_default_74 = fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:256 in forward, code: attention_scores = attention_scores.view(
        reshape_default_1: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_self, [192, 512, 512]);  where_self = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([192, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 24, 512, 512], dtype=torch.bool, device='cuda'),
    torch.randn([8, 24, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
