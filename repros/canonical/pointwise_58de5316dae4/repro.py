"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f16[2048, 768]", addmm_1: "f16[2048, 768]", addmm_2: "f16[2048, 768]", arg9_1: "b8[4, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm, [4, 512, 768]);  addmm = None
        mul_tensor: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(reshape_default, 0.125);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:152 in forward, code: query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_1: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(mul_tensor, [4, -1, 12, 64]);  mul_tensor = None
        permute_default: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        reshape_default_2: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:156 in forward, code: key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_3: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_2, [4, -1, 12, 64]);  reshape_default_2 = None
        permute_default_1: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        reshape_default_4: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 768]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:157 in forward, code: value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        reshape_default_5: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_4, [4, -1, 12, 64]);  reshape_default_4 = None
        permute_default_2: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f16[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f16[4, 1, 512, 512]" = torch.ops.aten.where.self(arg9_1, full_default, full_default_1);  arg9_1 = full_default = full_default_1 = None
        return (permute_default, permute_default_1, permute_default_2, where_self)


def _default_make_inputs():
    return [
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1, 512, 512], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
