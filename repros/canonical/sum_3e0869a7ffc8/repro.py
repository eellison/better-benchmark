"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_training
Pattern hash: 3e0869a7ffc8
Shape hash: ee7f3808
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_69: "f32[96, 512, 512]", _shape_param_0, gt_1: "b8[8, 12, 512, 512]", bmm: "f32[96, 512, 512]", _shape_param_1, amax_default_11: "f32[8, 12, 512, 1]", sum_1: "f32[8, 12, 512, 1]", _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        reshape_default: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, _shape_param_0);  bmm_69 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_default: "f32[8, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        reshape_default_1: "f32[8, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None

        # No stacktrace found for following nodes
        mul_tensor_2: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default_1, 1);  reshape_default_1 = None
        sub_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_11);  mul_tensor_2 = amax_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        mul_tensor_3: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        exp_default: "f32[8, 12, 512, 512]" = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        div_tensor: "f32[8, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        mul_tensor_4: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [-1], True)
        neg_default: "f32[8, 12, 512, 512]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[8, 12, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_4);  neg_default = sum_dim_int_list = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_tensor_5: "f32[8, 12, 512, 512]" = torch.ops.aten.mul.Tensor(fma_default, 0.125);  fma_default = None
        reshape_default_2: "f32[96, 512, 512]" = torch.ops.aten.reshape.default(mul_tensor_5, _shape_param_2);  mul_tensor_5 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([96, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 12, 512, 512],  # _shape_param_0
    torch.randint(0, 2, [8, 12, 512, 512], dtype=torch.bool, device='cuda'),
    torch.randn([96, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 12, 512, 512],  # _shape_param_1
    torch.randn([8, 12, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 512, 1], dtype=torch.float32, device='cuda'),
    [96, 512, 512],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
