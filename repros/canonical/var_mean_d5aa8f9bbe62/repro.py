"""
Standalone repro captured via capture_hook.
Label: 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_infer
Pattern hash: d5aa8f9bbe62
Shape hash: 0c0c5443
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 128, 80], f32), T([2560, 2560], f32), T([8008, 2560], f32), T([16, 128], i64, gen=Index(8008)), T([128, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), S([16, 32, 128, 80]), S([16, 128, -1]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]), S([2048, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_3: "f32[512, 128, 80]", arg28_1: "f32[2560, 2560]", arg2_1: "f32[8008, 2560]", arg1_1: "i64[16, 128]", arg38_1: "f32[128, 2560]", arg39_1: "f32[2560]", arg40_1: "f32[2560]", arg41_1: "f32[2560, 2560]", arg43_1: "f32[2560, 2560]", arg45_1: "f32[2560, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_1: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default_2: "f32[2048, 2560]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[16, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg2_1 = arg1_1 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:585 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_default_1: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg38_1, add_tensor);  arg38_1 = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:616 in forward, code: hidden_states = inputs_embeds + position_ids
        add_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg39_1);  mul_tensor_1 = arg39_1 = None
        add_tensor_3: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg40_1);  mul_tensor_2 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_3);  _shape_param_3 = None
        permute_default_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_4: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_4);  _shape_param_4 = None
        permute_default_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_5: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_3, _shape_param_5);  add_tensor_3 = _shape_param_5 = None
        permute_default_4: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        return (reshape_default_2, permute_default_1, reshape_default_3, permute_default_2, reshape_default_4, permute_default_3, reshape_default_5, permute_default_4)


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
