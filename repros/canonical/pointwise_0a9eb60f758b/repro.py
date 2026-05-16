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
    def forward(self, getitem_90: "f32[8, 16, 1024, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:106 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:271 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        reshape_default: "f32[8, 1024, 1024]" = torch.ops.aten.reshape.default(permute_default, [8, 1024, -1]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:272 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(reshape_default, [8192, 1024]);  reshape_default = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([8, 16, 1024, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
