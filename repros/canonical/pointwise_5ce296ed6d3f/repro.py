"""
Standalone repro captured via capture_hook.
Label: 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_inference
Pattern hash: 5ce296ed6d3f
Shape hash: e32ee600
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
    def forward(self, add_29: "f32[8, 128, 2560]", getitem_17: "f32[8, 128, 1]", getitem_16: "f32[8, 128, 1]", arg48_1: "f32[2560]", arg49_1: "f32[2560]", _shape_param_0, arg50_1: "f32[2560, 2560]", add_18: "f32[8, 128, 2560]", getitem_9: "f32[8, 128, 1]", getitem_8: "f32[8, 128, 1]", arg35_1: "f32[2560]", arg36_1: "f32[2560]", _shape_param_1, arg52_1: "f32[2560, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        sub_tensor: "f32[8, 128, 2560]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  add_29 = getitem_17 = None
        add_tensor: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_default: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg48_1);  mul_tensor = arg48_1 = None
        add_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg49_1);  mul_tensor_1 = arg49_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[1024, 2560]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
        add_tensor_2: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_default_1: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_2: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[8, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg35_1);  mul_tensor_2 = arg35_1 = None
        add_tensor_3: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg36_1);  mul_tensor_3 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[1024, 2560]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_0
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    torch.randn([2560], dtype=torch.float32, device='cuda'),
    [1024, 2560],  # _shape_param_1
    torch.randn([2560, 2560], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
