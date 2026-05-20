"""
Standalone repro captured via capture_hook.
Label: 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_infer
Pattern hash: 76165e3233a7
Shape hash: 1b943457
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8008, 2560], f32), T([16, 128], i64, max=8008), T([128, 2560], f32), T([2560], f32), T([2560], f32), T([2560, 2560], f32), T([2560, 2560], f32), S([2048, 2560]), S([2048, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[8008, 2560]", arg1_1: "i64[16, 128]", arg3_1: "f32[128, 2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560]", arg6_1: "f32[2560, 2560]", arg8_1: "f32[2560, 2560]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f32[16, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg2_1 = arg1_1 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:79 in forward, code: position_ids = torch.arange(
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_default_1: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg3_1, iota_default);  arg3_1 = iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:484 in forward, code: hidden_states = inputs_embeds + embed_pos
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default_1: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1)


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
