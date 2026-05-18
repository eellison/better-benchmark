"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1536'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_142: "f32[192, 64, 512]", full_default_1: "f32[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:247 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        div_tensor: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_142, full_default_1);  bmm_142 = full_default_1 = None
        permute_default: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:193 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_default, [8, 24, 512, 64]);  permute_default = None
        permute_default_1: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:192 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_1: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_default_1, [8, 512, 1536]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:236 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_default: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(reshape_default_1, memory_format = torch.contiguous_format);  reshape_default_1 = None
        reshape_default_2: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_default, [4096, 1536]);  clone_default = None
        permute_default_2: "f32[1536, 4096]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1536]);  sum_dim_int_list = None
        return (permute_default_2, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([192, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
