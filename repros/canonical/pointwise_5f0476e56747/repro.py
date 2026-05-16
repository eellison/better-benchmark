"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_inference
Pattern hash: 5f0476e56747
Shape hash: ef90ccd4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_70: "f32[1024, 4096]", _shape_param_0, _shape_param_1, arg193_1: "f32[1024, 4096]", getitem_52: "f32[8, 16, 128, 64]", _shape_param_2, _shape_param_3, arg206_1: "f32[1024, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:373 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[8, 128, 4096]" = torch.ops.aten.reshape.default(addmm_70, _shape_param_0);  addmm_70 = _shape_param_0 = None
        relu_default: "f32[8, 128, 4096]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:375 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[1024, 4096]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_1: "f32[8, 128, 16, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_2);  permute_default_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_3: "f32[1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None
        permute_default_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        return (reshape_default_1, permute_default, reshape_default_3, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [8, 128, 4096],  # _shape_param_0
    [1024, 4096],  # _shape_param_1
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    torch.randn(1048576, dtype=torch.float32, device='cuda').as_strided([8, 16, 128, 64], [131072, 64, 1024, 1]),  # getitem_52
    [8, 128, -1],  # _shape_param_2
    [1024, 1024],  # _shape_param_3
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
