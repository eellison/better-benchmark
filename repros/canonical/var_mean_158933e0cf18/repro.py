"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_infer
Pattern hash: 158933e0cf18
Shape hash: 5fee0a78
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 768], f32), T([8192, 768], f32), T([768], f32), T([768], f32), T([768, 768], f32), T([768, 768], f32), T([768, 768], f32), S([4, 2048, 768]), S([8192, 768]), S([8192, 768]), S([8192, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_153: "f32[8192, 768]", addmm_65: "f32[8192, 768]", arg179_1: "f32[768]", arg180_1: "f32[768]", arg181_1: "f32[768, 768]", arg183_1: "f32[768, 768]", arg185_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:247 in forward, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_tensor: "f32[8192, 768]" = torch.ops.aten.add.Tensor(view_153, addmm_65);  view_153 = addmm_65 = None
        reshape_default: "f32[4, 2048, 768]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_0);  add_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(reshape_default, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 2048, 1]" = var_mean_correction[0]
        getitem_1: "f32[4, 2048, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[4, 2048, 768]" = torch.ops.aten.sub.Tensor(reshape_default, getitem_1);  reshape_default = getitem_1 = None
        add_tensor_1: "f32[4, 2048, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg179_1);  mul_tensor = arg179_1 = None
        add_tensor_2: "f32[4, 2048, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg180_1);  mul_tensor_1 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        reshape_default_2: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_2);  _shape_param_2 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        reshape_default_3: "f32[8192, 768]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1, reshape_default_3, permute_default_2)


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
