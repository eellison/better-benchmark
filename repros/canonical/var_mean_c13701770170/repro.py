"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer
Pattern hash: c13701770170
Shape hash: fa5e91d1
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
_shapes_config = "(T([12000, 384], f16), T([8, 1500, 384], f16, stride=(576000, 1, 1500)), T([384], f16), T([384], f16), S([8, 1500, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_17: "f16[12000, 384]", convert_element_type_78: "f16[8, 1500, 384]", arg60_1: "f16[384]", arg61_1: "f16[384]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:354 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f16[8, 1500, 384]" = torch.ops.aten.reshape.default(addmm_17, _shape_param_0);  addmm_17 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:399 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f16[8, 1500, 384]" = torch.ops.aten.add.Tensor(convert_element_type_78, reshape_default);  convert_element_type_78 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:402 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        clone_default: "f16[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        convert_element_type_default: "f32[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_1: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg60_1);  mul_tensor = arg60_1 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg61_1);  mul_tensor_1 = arg61_1 = None
        convert_element_type_default_1: "f16[8, 1500, 384]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:403 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default_1: "f16[12000, 384]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return reshape_default_1



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
