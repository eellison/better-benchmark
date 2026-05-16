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
    def forward(self, getitem_52: "f32[32, 6, 512, 64]", bmm_33: "f32[98304, 64, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:401 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[32, 512, 6, 64]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None
        clone_default: "f32[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:380 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        reshape_default: "f32[16384, 384]" = torch.ops.aten.reshape.default(bmm_33, [-1, 384]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:403 in forward, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        reshape_default_1: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, [32, -1, 6, 64]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:404 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        cat_default: "f32[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone_default, reshape_default_1], 2);  clone_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:410 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        reshape_default_2: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(cat_default, [32, 512, 768]);  cat_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:424 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default_3: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_2, [16384, 768]);  reshape_default_2 = None
        return reshape_default_3


def _default_make_inputs():
    return [
    torch.randn([32, 6, 512, 64], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 64, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
