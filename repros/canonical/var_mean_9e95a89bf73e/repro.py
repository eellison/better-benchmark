"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_infer
Pattern hash: 9e95a89bf73e
Shape hash: 45c5bba0
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50265, 768], f16), T([1, 512], i64, gen=Index(50265)), T([1026, 768], f16), T([768], f16), T([768], f16), S([512, 768]), S([512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[50265, 768]", arg0_1: "i64[1, 512]", arg3_1: "f16[1026, 768]", arg4_1: "f16[768]", arg5_1: "f16[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_default: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg1_1 = arg0_1 = None
        mul_tensor: "f16[1, 512, 768]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:621 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:96 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:98 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor_1: "i64[1, 512]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default_1: "f16[1, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:651 in forward, code: hidden_states = inputs_embeds + positions
        add_tensor_2: "f16[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1);  mul_tensor = embedding_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:652 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_3: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        add_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        convert_element_type_default_1: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float16);  add_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:207 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_0);  _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:227 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_1: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:228 in forward, code: value_states = self.v_proj(current_states)
        reshape_default_2: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        return (reshape_default, reshape_default_1, reshape_default_2)



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
