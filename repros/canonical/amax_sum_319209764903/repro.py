"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_inference
Pattern hash: 319209764903
Shape hash: 1590a9e2
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[128, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, addmm_140: "f32[4096, 1024]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:171 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default: "f32[8, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 1);  reshape_default = None
        amax_default: "f32[8, 16, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:179 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        div_tensor: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[8, 16, 512, 512]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_dim_int_list: "f32[8, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[8, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_default: "f32[8, 16, 512, 512]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_1);  div_tensor_1 = _shape_param_1 = None
        reshape_default_1: "f32[128, 512, 512]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        reshape_default_2: "f32[8, 512, 1024]" = torch.ops.aten.reshape.default(addmm_140, _shape_param_3);  addmm_140 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 512, 16, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default: "f32[8, 16, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_default_1: "f32[8, 16, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 16, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[128, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (reshape_default_1, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([128, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 16, 512, 512],  # _shape_param_0
    [8, 16, 512, 512],  # _shape_param_1
    [128, 512, 512],  # _shape_param_2
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [8, 512, 1024],  # _shape_param_3
    [8, 512, -1, 64],  # _shape_param_4
    [8, 16, 512, 64],  # _shape_param_5
    [128, 512, 64],  # _shape_param_6
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
