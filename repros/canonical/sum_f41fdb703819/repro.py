"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: f41fdb703819
Shape hash: cf8de827
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([192, 512, 512], f32), T([8, 24, 512, 512], b8), T([8, 24, 512, 512], f32), T([], f32), S([8, 24, 512, 512]), S([192, 512, 512]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_49: "f32[192, 512, 512]", gt_70: "b8[8, 24, 512, 512]", div_47: "f32[8, 24, 512, 512]", full_default_74: "f32[]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        reshape_default: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, _shape_param_0);  bmm_49 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_default: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_tensor: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_tensor_2: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_47);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
        neg_default: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_default: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default, full_default_74, fma_default);  full_default = full_default_74 = fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        reshape_default_1: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        return reshape_default_1


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
