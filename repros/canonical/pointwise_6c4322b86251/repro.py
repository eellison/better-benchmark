"""
Standalone repro captured via capture_hook.
Label: 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_inference
Pattern hash: 6c4322b86251
Shape hash: 0ceadf73
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
    def forward(self, addmm_10: "f32[1024, 10240]", _shape_param_0, _shape_param_1, arg33_1: "f32[2560, 10240]", getitem_12: "f32[8, 32, 128, 80]", _shape_param_2, _shape_param_3, arg46_1: "f32[2560, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[8, 128, 10240]" = torch.ops.aten.reshape.default(addmm_10, _shape_param_0);  addmm_10 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_tensor: "f32[8, 128, 10240]" = torch.ops.aten.mul.Tensor(reshape_default, 0.5)
        mul_tensor_1: "f32[8, 128, 10240]" = torch.ops.aten.mul.Tensor(reshape_default, 0.7071067811865476);  reshape_default = None
        erf_default: "f32[8, 128, 10240]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[8, 128, 10240]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[8, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[1024, 10240]" = torch.ops.aten.reshape.default(mul_tensor_2, _shape_param_1);  mul_tensor_2 = _shape_param_1 = None
        permute_default: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f32[8, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[1024, 2560]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        return (reshape_default_1, permute_default, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([1024, 10240], dtype=torch.float32, device='cuda'),
    [8, 128, 10240],  # _shape_param_0
    [1024, 10240],  # _shape_param_1
    torch.randn([2560, 10240], dtype=torch.float32, device='cuda'),
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([8, 32, 128, 80], [327680, 80, 2560, 1]),  # getitem_12
    [8, 128, -1],  # _shape_param_2
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
