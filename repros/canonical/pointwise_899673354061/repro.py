"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 899673354061
Shape hash: 19c4bafe
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 128100], f32), T([192, 64, 512], f32), T([], f32), T([192, 512, 64], f32), T([1536, 1536], f32), T([1536, 1536], f32), T([192, 512, 64], f32), T([1536, 1536], f32), S([8, 24, 512, 64]), S([8, 512, 1536]), S([4096, 1536]), S([8, 24, 512, 64]), S([8, 512, 1536]), S([4096, 1536]), S([8, 24, 512, 64]), S([8, 512, 1536]), S([4096, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, view_535: "f32[4096, 128100]", bmm_142: "f32[192, 64, 512]", full_default_1: "f32[]", bmm_140: "f32[192, 512, 64]", primals_11: "f32[1536, 1536]", primals_9: "f32[1536, 1536]", bmm_143: "f32[192, 512, 64]", primals_7: "f32[1536, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_default: "f32[128100, 4096]" = torch.ops.aten.permute.default(view_535, [1, 0]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        div_tensor: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_142, full_default_1);  bmm_142 = full_default_1 = None
        permute_default_1: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None
        permute_default_2: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_default: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_1: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        reshape_default_2: "f32[4096, 1536]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_3: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_default_4: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default_3: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        permute_default_5: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        reshape_default_4: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_default_5, _shape_param_4);  permute_default_5 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_default_1: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_5: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        permute_default_6: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_default_7: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_6, [1, 0]);  permute_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        reshape_default_6: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_6);  bmm_143 = _shape_param_6 = None
        permute_default_8: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1, 3]);  reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_default_2: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        reshape_default_7: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        reshape_default_8: "f32[4096, 1536]" = torch.ops.aten.reshape.default(reshape_default_7, _shape_param_8);  reshape_default_7 = _shape_param_8 = None
        permute_default_9: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_10: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None
        return (permute_default, reshape_default_2, permute_default_4, reshape_default_5, permute_default_7, reshape_default_8, permute_default_10)


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
