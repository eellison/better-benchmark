"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_training
Pattern hash: f4158dd9dd08
Shape hash: b5a04b51
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_138: "f32[192, 64, 512]", full_default_1: "f32[]", bmm_136: "f32[192, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, primals_27: "f32[1536, 1536]", _shape_param_3, _shape_param_4, _shape_param_5, primals_25: "f32[1536, 1536]", bmm_139: "f32[192, 512, 64]", _shape_param_6, _shape_param_7, _shape_param_8, primals_23: "f32[1536, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        div_tensor: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_138, full_default_1);  bmm_138 = full_default_1 = None
        permute_default: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_136, _shape_param_0);  bmm_136 = _shape_param_0 = None
        permute_default_1: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_default: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_1: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        reshape_default_2: "f32[4096, 1536]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_2: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_default_3: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default_3: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_default, _shape_param_3);  permute_default = _shape_param_3 = None
        permute_default_4: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_4: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_default_1: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_5: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_default_6: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default_6: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_139, _shape_param_6);  bmm_139 = _shape_param_6 = None
        permute_default_7: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_default_2: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        reshape_default_7: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        reshape_default_8: "f32[4096, 1536]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_8: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_default_9: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_8, [1, 0]);  permute_default_8 = None
        return (reshape_default_2, permute_default_3, reshape_default_5, permute_default_6, reshape_default_8, permute_default_9)


def _default_make_inputs():
    return [
    torch.randn([192, 64, 512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 24, 512, 64],  # _shape_param_0
    [8, 512, 1536],  # _shape_param_1
    [4096, 1536],  # _shape_param_2
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    [8, 24, 512, 64],  # _shape_param_3
    [8, 512, 1536],  # _shape_param_4
    [4096, 1536],  # _shape_param_5
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    torch.randn([192, 512, 64], dtype=torch.float32, device='cuda'),
    [8, 24, 512, 64],  # _shape_param_6
    [8, 512, 1536],  # _shape_param_7
    [4096, 1536],  # _shape_param_8
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
