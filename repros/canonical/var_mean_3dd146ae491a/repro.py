"""
Standalone repro captured via capture_hook.
Label: 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_inference
Pattern hash: 3dd146ae491a
Shape hash: 10164687
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_9: "f32[1024, 2560]", _shape_param_0, add_10: "f32[8, 128, 2560]", add_26: "f32[8, 128, 2560]", getitem_11: "f32[8, 128, 1]", getitem_10: "f32[8, 128, 1]", arg38_1: "f32[2560]", arg39_1: "f32[2560]", _shape_param_1, arg40_1: "f32[2560, 2560]", _shape_param_2, arg42_1: "f32[2560, 2560]", _shape_param_3, arg44_1: "f32[2560, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(addmm_9, _shape_param_0);  addmm_9 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, reshape_default);  add_10 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[8, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 128, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_tensor: "f32[8, 128, 2560]" = torch.ops.aten.sub.Tensor(add_26, getitem_11);  add_26 = getitem_11 = None
        add_tensor_1: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_default: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg38_1);  mul_tensor = arg38_1 = None
        add_tensor_2: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg39_1);  mul_tensor_1 = arg39_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[1024, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[1024, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_3: "f32[1024, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 2560], dtype=torch.float32, device='cuda'),
    [8, 128, 2560],  # _shape_param_0
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_1
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_2
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_3
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
