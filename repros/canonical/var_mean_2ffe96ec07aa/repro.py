"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_distil_whisper_infer
Pattern hash: 2ffe96ec07aa
Shape hash: f1c44b4d
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
_shapes_config = "(T([1500, 1024], f16), T([1, 1500, 1024], f16, stride=(1536000, 1, 1500)), T([1024], f16), T([1024], f16), S([1, 1500, 1024]), S([1500, 1024]), S([1500, 1024]), S([1500, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_109: "f16[1500, 1024]", add_152: "f16[1, 1500, 1024]", arg336_1: "f16[1024]", arg337_1: "f16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default: "f16[1, 1500, 1024]" = torch.ops.aten.reshape.default(addmm_109, _shape_param_0);  addmm_109 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:407 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[1, 1500, 1024]" = torch.ops.aten.add.Tensor(add_152, reshape_default);  add_152 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:411 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_default: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        clamp_min_default: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_min.default(convert_element_type_default, -64504.0);  convert_element_type_default = None
        clamp_max_default: "f32[1, 1500, 1024]" = torch.ops.aten.clamp_max.default(clamp_min_default, 64504.0);  clamp_min_default = None
        convert_element_type_default_1: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clamp_max_default, torch.float16);  clamp_max_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:392 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        clone_default: "f16[1, 1500, 1024]" = torch.ops.aten.clone.default(convert_element_type_default_1, memory_format = torch.contiguous_format);  convert_element_type_default_1 = None
        convert_element_type_default_2: "f32[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 1500, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_1: "f32[1, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 1500, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg336_1);  mul_tensor = arg336_1 = None
        add_tensor_2: "f32[1, 1500, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg337_1);  mul_tensor_1 = arg337_1 = None
        convert_element_type_default_3: "f16[1, 1500, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:309 in forward, code: query_states = (self.q_proj(hidden_states) * self.scaling).view(hidden_shape).transpose(1, 2).contiguous()
        reshape_default_1: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_1);  _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:332 in forward, code: key_states = self.k_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_2: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_2);  _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:333 in forward, code: value_states = self.v_proj(current_states).view(kv_shape).transpose(1, 2).contiguous()
        reshape_default_3: "f16[1500, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
        return (reshape_default_2, reshape_default_3, reshape_default_1)



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
