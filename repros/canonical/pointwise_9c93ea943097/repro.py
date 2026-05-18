"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_inference
Pattern hash: 9c93ea943097
Shape hash: 4e50c7ab
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_138: "f32[4096, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, addmm_139: "f32[4096, 1536]", _shape_param_3, _shape_param_4, _shape_param_5, add_163: "f32[8, 512, 1536]", _shape_param_6, arg378_1: "f32[1536, 1536]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        reshape_default: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_138, _shape_param_0);  addmm_138 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_1: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_default: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None
        clone_default: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        reshape_default_3: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_139, _shape_param_3);  addmm_139 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_4: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        permute_default_1: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        clone_default_1: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_5: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        permute_default_2: "f32[192, 64, 512]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        full_default: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        div_tensor: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_default_2, full_default);  permute_default_2 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        reshape_default_6: "f32[4096, 1536]" = torch.ops.aten.reshape.default(add_163, _shape_param_6);  add_163 = _shape_param_6 = None
        permute_default_3: "f32[1536, 1536]" = torch.ops.aten.permute.default(arg378_1, [1, 0]);  arg378_1 = None
        return (reshape_default_2, div_tensor, reshape_default_6, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([4096, 1536], dtype=torch.float32, device='cuda'),
    [8, 512, 1536],  # _shape_param_0
    [8, 512, 24, -1],  # _shape_param_1
    [-1, 512, 64],  # _shape_param_2
    torch.randn([4096, 1536], dtype=torch.float32, device='cuda'),
    [8, 512, 1536],  # _shape_param_3
    [8, 512, 24, -1],  # _shape_param_4
    [-1, 512, 64],  # _shape_param_5
    torch.randn([8, 512, 1536], dtype=torch.float32, device='cuda'),
    [4096, 1536],  # _shape_param_6
    torch.randn([1536, 1536], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
