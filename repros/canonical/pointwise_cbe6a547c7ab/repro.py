"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: cbe6a547c7ab
Shape hash: c187c0cd
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 4096], f32), T([1024, 4096], f32), T([64, 16, 128, 64], f32, stride=(131072, 64, 1024, 1)), T([1024, 1024], f32), S([64, 128, 4096]), S([8192, 4096]), S([64, 128, -1]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_70: "f32[8192, 4096]", arg194_1: "f32[1024, 4096]", getitem_52: "f32[64, 16, 128, 64]", arg207_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[64, 128, 4096]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_0);  addmm_70 = _shape_param_0 = None
        relu_default: "f32[64, 128, 4096]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[8192, 4096]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        return (reshape_default_1, permute_default, reshape_default_3, permute_default_2)


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
