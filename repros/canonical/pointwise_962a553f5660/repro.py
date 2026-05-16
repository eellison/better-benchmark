"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[2048, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:315 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 512]);  addmm_2 = None
        reshape_default_1: "f32[4, 512, 8, 64]" = torch.ops.aten.reshape.default(reshape_default, [4, 512, -1, 64]);  reshape_default = None
        permute_default: "f32[4, 8, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:218 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_default: "f32[4, 8, 1, 512, 64]" = torch.ops.aten.unsqueeze.default(permute_default, 2);  permute_default = None
        expand_default: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.expand.default(unsqueeze_default, [4, 8, 8, 512, 64]);  unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:219 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_default: "f32[4, 8, 8, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[4, 64, 512, 64]" = torch.ops.aten.reshape.default(clone_default, [4, 64, 512, 64]);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:268 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f32[4, 64, 512, 64]" = torch.ops.aten.expand.default(reshape_default_2, [4, 64, 512, 64]);  reshape_default_2 = None
        reshape_default_3: "f32[256, 512, 64]" = torch.ops.aten.reshape.default(expand_default_1, [256, 512, 64]);  expand_default_1 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
