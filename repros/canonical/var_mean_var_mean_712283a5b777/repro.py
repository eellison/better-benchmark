"""
Standalone repro captured via capture_hook.
Label: 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_inference
Pattern hash: 712283a5b777
Shape hash: 9a702c34
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
    def forward(self, addmm_11: "f32[1024, 2560]", _shape_param_0, add_14: "f32[8, 128, 2560]", addmm_15: "f32[1024, 2560]", _shape_param_1, add_26: "f32[8, 128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(addmm_11, _shape_param_0);  addmm_11 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(add_14, reshape_default);  add_14 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True);  add_tensor = None
        getitem: "f32[8, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 128, 1]" = var_mean_correction[1];  var_mean_correction = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[8, 128, 2560]" = torch.ops.aten.reshape.default(addmm_15, _shape_param_1);  addmm_15 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "f32[8, 128, 2560]" = torch.ops.aten.add.Tensor(add_26, reshape_default_1);  add_26 = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True);  add_tensor_1 = None
        getitem_2: "f32[8, 128, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[8, 128, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        return (getitem, getitem_1, getitem_2, getitem_3)


def _default_make_inputs():
    return [
    torch.randn([1024, 2560], dtype=torch.float32, device='cuda'),
    [8, 128, 2560],  # _shape_param_0
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 2560], dtype=torch.float32, device='cuda'),
    [8, 128, 2560],  # _shape_param_1
    torch.randn([8, 128, 2560], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
