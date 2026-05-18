"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_training
Pattern hash: b629c28d4d1f
Shape hash: a874d029
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_110: "f32[128, 4096]", _shape_param_0, _shape_param_1, bmm_54: "f32[16, 128, 128]", _shape_param_2, where: "f32[1, 1, 128, 128]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, primals_307: "f32[16384, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        reshape_default: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_110, _shape_param_0);  mm_110 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_1: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_2: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_54, _shape_param_2);  bmm_54 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(reshape_default_2, 16.0);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        add_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_tensor, where);  div_tensor = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_default: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        expand_default: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_3);  div_tensor_1 = _shape_param_3 = None
        reshape_default_3: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        reshape_default_4: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_6);  expand_default_1 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_1: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        return (reshape_default_3, reshape_default_4, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    [1, 128, 4096],  # _shape_param_0
    [1, 128, 16, 256],  # _shape_param_1
    torch.randn([16, 128, 128], dtype=torch.float32, device='cuda'),
    [1, 16, 128, 128],  # _shape_param_2
    torch.randn([1, 1, 128, 128], dtype=torch.float32, device='cuda'),
    [1, 16, 128, 128],  # _shape_param_3
    [16, 128, 128],  # _shape_param_4
    [1, 16, 128, 256],  # _shape_param_5
    [16, 128, 256],  # _shape_param_6
    torch.randn([16384, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
