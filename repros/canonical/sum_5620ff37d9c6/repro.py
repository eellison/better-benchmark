"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_training
Pattern hash: 5620ff37d9c6
Shape hash: 4e94f888
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_141: "f32[128, 512, 512]", _shape_param_0, gt_1: "b8[8, 16, 512, 512]", bmm: "f32[128, 512, 512]", _shape_param_1, amax_default_23: "f32[8, 16, 512, 1]", sum_1: "f32[8, 16, 512, 1]", _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        reshape_default: "f32[8, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_141, _shape_param_0);  bmm_141 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:183 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_default: "f32[8, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, mul_tensor);  reshape_default = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default_1: "f32[8, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm, _shape_param_1);  bmm = _shape_param_1 = None

        # No stacktrace found for following nodes
        mul_tensor_2: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default_1, 1);  reshape_default_1 = None
        sub_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_23);  mul_tensor_2 = amax_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        div_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[8, 16, 512, 512]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        div_tensor_1: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_1);  exp_default = sum_1 = None
        mul_tensor_3: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor_1);  mul_tensor_1 = None
        sum_dim_int_list: "f32[8, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [-1], True)
        neg_default: "f32[8, 16, 512, 512]" = torch.ops.aten.neg.default(div_tensor_1);  div_tensor_1 = None
        fma_default: "f32[8, 16, 512, 512]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_3);  neg_default = sum_dim_int_list = mul_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:173 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_tensor_2: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(fma_default, 8.0);  fma_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default_2: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(div_tensor_2, _shape_param_2);  div_tensor_2 = _shape_param_2 = None
        return reshape_default_2


def _default_make_inputs():
    return [
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 16, 512, 512],  # _shape_param_0
    torch.randint(0, 2, [8, 16, 512, 512], dtype=torch.bool, device='cuda'),
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 16, 512, 512],  # _shape_param_1
    torch.randn([8, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 16, 512, 1], dtype=torch.float32, device='cuda'),
    [128, 512, 512],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
