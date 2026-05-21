"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train
Pattern hash: 2a64dc79714d
Shape hash: ba34bbd0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 768], f32), T([4, 512, 768], f32), T([2048, 768], f32), T([2048, 768], f32), S([4, 512, 768]), S([4, 512, 768]), S([4, 512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_14: "f32[2048, 768]", mul_58: "f32[4, 512, 768]", mm_16: "f32[2048, 768]", mm_18: "f32[2048, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        reshape_default: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_14, _shape_param_0);  mm_14 = _shape_param_0 = None
        add_tensor: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, reshape_default);  mul_58 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_16, _shape_param_1);  mm_16 = _shape_param_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_18, _shape_param_2);  mm_18 = _shape_param_2 = None
        add_tensor_2: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None
        return add_tensor_2



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
