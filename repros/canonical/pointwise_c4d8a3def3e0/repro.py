"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: c4d8a3def3e0
Shape hash: 3d25c3cf
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 50257], f32), T([512, 128, 128], f32), T([512, 128, 128], f32), T([512, 128, 128], f32), T([2048, 2048], f32), T([2048, 2048], f32), T([2048, 2048], f32), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([32, 128, 2048]), S([32, 128, 2048]), S([32, 128, 2048]), S([4096, 2048]), S([4096, 2048]), S([4096, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, view_534: "f32[4096, 50257]", bmm_140: "f32[512, 128, 128]", bmm_142: "f32[512, 128, 128]", bmm_143: "f32[512, 128, 128]", primals_8: "f32[2048, 2048]", primals_7: "f32[2048, 2048]", primals_6: "f32[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[50257, 4096]" = torch.ops.aten.permute.default(view_534, [1, 0]);  view_534 = None
        constant_pad_nd_default: "f32[50260, 4096]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        reshape_default: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        reshape_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_1);  bmm_142 = _shape_param_1 = None
        reshape_default_2: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_2);  bmm_143 = _shape_param_2 = None
        permute_default_1: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_2: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_default: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        reshape_default_3: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_3: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_default_1, [0, 2, 1, 3]);  permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        reshape_default_4: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_4);  permute_default_3 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_default_4: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_default_1: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_5: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        reshape_default_6: "f32[4096, 2048]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_6);  reshape_default_3 = _shape_param_6 = None
        permute_default_5: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_6: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_5, [1, 0]);  permute_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_default_2: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(reshape_default_4, memory_format = torch.contiguous_format);  reshape_default_4 = None
        reshape_default_7: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_7);  clone_default_2 = _shape_param_7 = None
        permute_default_7: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_default_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_7, [1, 0]);  permute_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        reshape_default_8: "f32[4096, 2048]" = torch.ops.aten.reshape.default(reshape_default_5, _shape_param_8);  reshape_default_5 = _shape_param_8 = None
        permute_default_9: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_default_10: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_default_9, [1, 0]);  permute_default_9 = None
        return (constant_pad_nd_default, reshape_default_6, permute_default_6, reshape_default_7, permute_default_8, reshape_default_8, permute_default_10)


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
