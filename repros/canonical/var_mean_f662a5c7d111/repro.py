"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: f662a5c7d111
Shape hash: e383ec90
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([64, 128, 1024], f32), T([1024], f32), T([1024], f32), T([1024, 1024], f32), T([1024, 1024], f32), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, primals_3: "f32[64, 128, 1024]", primals_1: "f32[1024]", primals_2: "f32[1024]", primals_4: "f32[1024, 1024]", primals_6: "f32[1024, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:362 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_correction = torch.ops.aten.var_mean.correction(primals_3, [2], correction = 0, keepdim = True)
        getitem: "f32[64, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[64, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 128, 1024]" = torch.ops.aten.sub.Tensor(primals_3, getitem_1);  primals_3 = getitem_1 = None
        mul_tensor: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, primals_1);  mul_tensor = primals_1 = None
        add_tensor_1: "f32[64, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, primals_2);  mul_tensor_1 = primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:276 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        permute_default: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:296 in forward, code: key_states = self.k_proj(current_states)
        permute_default_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        return (reshape_default, permute_default, permute_default_1)


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
