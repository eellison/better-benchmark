"""
Standalone repro captured via capture_hook.
Label: pythia_410m
Pattern hash: 65633e5584ac
Shape hash: 530df824
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
    def forward(self, addmm_95: "f16[2048, 1024]", _shape_param_0, addmm_93: "f16[2048, 1024]", _shape_param_1, add_209: "f16[4, 512, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:48 in forward, code: hidden_states = self.dense_4h_to_h(hidden_states)
        reshape_default: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_95, _shape_param_0);  addmm_95 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:242 in forward, code: attn_output = self.dense(attn_output)
        reshape_default_1: "f16[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_93, _shape_param_1);  addmm_93 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:282 in forward, code: hidden_states = mlp_output + attn_output + hidden_states
        add_tensor: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        add_tensor_1: "f16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_tensor, add_209);  add_tensor = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neox/modeling_gpt_neox.py:376 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True);  convert_element_type_default = None
        getitem: "f32[4, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        _output_to_half_0: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem, torch.float16);  getitem = None
        _output_to_half_1: "f16[4, 512, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float16);  getitem_1 = None
        return (_output_to_half_0, _output_to_half_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_0
    torch.randn([2048, 1024], dtype=torch.float16, device='cuda'),
    [4, 512, 1024],  # _shape_param_1
    torch.randn([4, 512, 1024], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
