"""
Standalone repro captured via capture_hook.
Label: torchbench_moondream_infer
Pattern hash: 915d42788c50
Shape hash: f0f53470
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
_shapes_config = "(T([512, 2048], f16), T([512, 2048], f16), T([1, 512, 2048], f16), T([2048], f16), T([2048], f16), T([51200, 2048], f16), S([1, 512, 2048]), S([1, 512, 2048]), S([512, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_141: "f16[512, 2048]", addmm_143: "f16[512, 2048]", add_186: "f16[1, 512, 2048]", arg339_1: "f16[2048]", arg340_1: "f16[2048]", arg341_1: "f16[51200, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:252 in forward, code: attn_output = self.dense(attn_output)
        reshape_default: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_141, _shape_param_0);  addmm_141 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:267 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f16[1, 512, 2048]" = torch.ops.aten.reshape.default(addmm_143, _shape_param_1);  addmm_143 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:305 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_tensor: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        add_tensor_1: "f16[1, 512, 2048]" = torch.ops.aten.add.Tensor(add_tensor, add_186);  add_tensor = add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:398 in forward, code: hidden_states = self.final_layernorm(hidden_states)
        convert_element_type_default: "f32[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 512, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        add_tensor_2: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg339_1);  mul_tensor = arg339_1 = None
        add_tensor_3: "f32[1, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg340_1);  mul_tensor_1 = arg340_1 = None
        convert_element_type_default_1: "f16[1, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/phi/modeling_phi.py:465 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        reshape_default_2: "f16[512, 2048]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_2);  convert_element_type_default_1 = _shape_param_2 = None
        permute_default: "f16[2048, 51200]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        return (reshape_default_2, permute_default)



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
