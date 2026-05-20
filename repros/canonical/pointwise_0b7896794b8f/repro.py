"""
Standalone repro captured via capture_hook.
Label: hf_LayoutLMForMaskedLM_train
Pattern hash: 0b7896794b8f
Shape hash: e045a23d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([16384, 768], f32), S([32, 512, 768]), S([32, 512, 12, 64]))"

class Repro(torch.nn.Module):
    def forward(self, mm_140: "f32[16384, 768]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_140, _shape_param_0);  mm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[32, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_default: "f32[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        return clone_default


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
