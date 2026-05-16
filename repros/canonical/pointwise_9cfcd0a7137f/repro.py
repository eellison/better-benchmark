"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_training
Pattern hash: 9cfcd0a7137f
Shape hash: a02aaf92
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[1024, 4096]", _shape_param_0, le: "b8[8, 128, 4096]", _shape_param_1, primals_27: "f32[4096, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f32[8, 128, 4096]" = torch.ops.aten.reshape.default(mm, _shape_param_0);  mm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        where_self: "f32[8, 128, 4096]" = torch.ops.aten.where.self(le, full_default, reshape_default);  le = full_default = reshape_default = None
        reshape_default_1: "f32[1024, 4096]" = torch.ops.aten.reshape.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        permute_default: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_default_1: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [8, 128, 4096],  # _shape_param_0
    torch.randint(0, 2, [8, 128, 4096], dtype=torch.bool, device='cuda'),
    [1024, 4096],  # _shape_param_1
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
