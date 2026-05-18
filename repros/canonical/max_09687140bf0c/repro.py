"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=max, ranges=[], reduction_ranges=[]
#   origins: ['aten.max.dim']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm: "f32[256, 512, 512]", arg11_1: "f32[4, 1, 512, 512]", arg10_1: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:253 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        reshape_default: "f32[4, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 64, 512, 512]);  bmm = None
        mul_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:256 in eager_attention_forward, code: attn_weights = attn_weights + causal_mask
        add_tensor: "f32[4, 64, 512, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg11_1);  mul_tensor = arg11_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:258 in eager_attention_forward, code: sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        reshape_default_1: "f32[1, 64, 1, 1]" = torch.ops.aten.reshape.default(arg10_1, [1, -1, 1, 1]);  arg10_1 = None
        expand_default: "f32[4, 64, 512, 1]" = torch.ops.aten.expand.default(reshape_default_1, [4, -1, 512, -1]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:259 in eager_attention_forward, code: combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        cat_default: "f32[4, 64, 512, 513]" = torch.ops.aten.cat.default([add_tensor, expand_default], -1);  add_tensor = expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py:264 in eager_attention_forward, code: combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        max_dim = torch.ops.aten.max.dim(cat_default, -1, True);  cat_default = None
        getitem: "f32[4, 64, 512, 1]" = max_dim[0];  max_dim = None
        return getitem


def _default_make_inputs():
    return [
    torch.randn([256, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
