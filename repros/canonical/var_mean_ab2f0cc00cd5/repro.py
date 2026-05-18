"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_inference
Pattern hash: ab2f0cc00cd5
Shape hash: e7f2aa6d
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
    def forward(self, bmm_23: "f32[128, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, arg187_1: "f32[1024, 1024]", arg1_1: "f32[128112, 1024]", arg0_1: "i64[8, 128]", cumsum_1: "i64[8, 128]", convert_element_type_3: "i32[8, 128]", arg197_1: "f32[1026, 1024]", _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, _shape_param_0);  bmm_23 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[8, 128, 16, 64]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[8, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:323 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[1024, 1024]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:77 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[8, 128, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f32[8, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:178 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        convert_element_type_default: "i32[8, 128]" = torch.ops.prims.convert_element_type.default(cumsum_1, torch.int32);  cumsum_1 = None
        add_tensor: "i32[8, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default, 0);  convert_element_type_default = None
        mul_tensor_1: "i32[8, 128]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type_3);  add_tensor = convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:179 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_default_1: "i64[8, 128]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.int64);  mul_tensor_1 = None
        add_tensor_1: "i64[8, 128]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:144 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
        reshape_default_3: "i64[1024]" = torch.ops.aten.reshape.default(add_tensor_1, [-1]);  add_tensor_1 = None
        index_tensor: "f32[1024, 1024]" = torch.ops.aten.index.Tensor(arg197_1, [reshape_default_3]);  arg197_1 = reshape_default_3 = None
        reshape_default_4: "f32[8, 128, 1024]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_3);  index_tensor = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:687 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_2: "f32[8, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor, reshape_default_4);  mul_tensor = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:441 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True);  add_tensor_2 = None
        getitem: "f32[8, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        return (reshape_default_2, permute_default_1, getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([128, 128, 64], dtype=torch.float32, device='cuda'),
    [8, 16, 128, 64],  # _shape_param_0
    [8, 128, -1],  # _shape_param_1
    [1024, 1024],  # _shape_param_2
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([128112, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 128112, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1026, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randint(0, 1026, [8, 128], dtype=torch.int32, device='cuda'),
    torch.randn([1026, 1024], dtype=torch.float32, device='cuda'),
    [8, 128, 1024],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
