"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer
Pattern hash: 576df1db0d25
Shape hash: 610f0f6a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([8192, 1024], f32), T([64, 128, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([64, 128, 1024], f32), T([1024, 1024], f32), S([64, 128, 1024]), S([8192, 1024]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_185: "f32[8192, 1024]", add_207: "f32[64, 128, 1024]", arg495_1: "f32[1024]", arg496_1: "f32[1024]", arg497_1: "f32[1024, 1024]", add_90: "f32[64, 128, 1024]", arg499_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:324 in forward, code: attn_output = self.out_proj(attn_output)
        reshape_default: "f32[64, 128, 1024]" = torch.ops.aten.reshape.default(addmm_185, _shape_param_0);  addmm_185 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:451 in forward, code: hidden_states = residual + hidden_states
        add_tensor: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(add_207, reshape_default);  add_207 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:456 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg495_1);  mul_tensor = arg495_1 = None
        add_tensor_2: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg496_1);  mul_tensor_1 = arg496_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        reshape_default_2: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_90, _shape_param_2);  add_90 = _shape_param_2 = None
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)


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
