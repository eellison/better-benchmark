"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Albert_train
Pattern hash: 145b47f2ec84
Shape hash: 56d66f04
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
_shapes_config = "(T([4096, 768], f32), T([8, 512, 768], f32), T([4096, 768], f32), T([4096, 768], f32), S([8, 512, 768]), S([8, 512, 768]), S([8, 512, 768]), S([4096, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm_142: "f32[4096, 768]", mul_453: "f32[8, 512, 768]", mm_144: "f32[4096, 768]", mm_146: "f32[4096, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_142, _shape_param_0);  mm_142 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(mul_453, reshape_default);  mul_453 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_144, _shape_param_1);  mm_144 = _shape_param_1 = None
        add_tensor_1: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(mm_146, _shape_param_2);  mm_146 = _shape_param_2 = None
        add_tensor_2: "f32[8, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:276 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        reshape_default_3: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        return reshape_default_3



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
