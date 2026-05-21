"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_distil_whisper_infer
Pattern hash: 0ed6022de1c0
Shape hash: ef5545f4
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
_shapes_config = "(T([1, 1024, 1500], f16), T([1500, 1024], f16), T([1024], f16), T([1024], f16), S([1500, 1024]), S([1500, 1024]), S([1500, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_1: "f16[1, 1024, 1500]", arg5_1: "f16[1500, 1024]", arg6_1: "f16[1024]", arg7_1: "f16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:619 in forward, code: inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        convert_element_type_default: "f32[1, 1024, 1500]" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        mul_tensor: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[1, 1024, 1500]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 1024, 1500]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 1024, 1500]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "f16[1, 1024, 1500]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.float16);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:621 in forward, code: inputs_embeds = inputs_embeds.permute(0, 2, 1)
        permute_default: "f16[1, 1500, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1]);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:622 in forward, code: all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        iota_default: "i64[1500]" = torch.ops.prims.iota.default(1500, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:624 in forward, code: hidden_states = inputs_embeds + self.embed_positions(all_positions)
        embedding_default: "f16[1500, 1024]" = torch.ops.aten.embedding.default(arg5_1, iota_default);  arg5_1 = iota_default = None
        add_tensor_1: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(permute_default, embedding_default);  permute_default = embedding_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_default: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format);  add_tensor_1 = None
        convert_element_type_default_2: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_2: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_3: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_4: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, arg6_1);  mul_tensor_3 = arg6_1 = None
        add_tensor_3: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, arg7_1);  mul_tensor_4 = arg7_1 = None
        convert_element_type_default_3: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        reshape_default: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_0);  _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_1: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_2: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_2);  convert_element_type_default_3 = _shape_param_2 = None
        return (reshape_default_2, reshape_default_1, reshape_default)



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
