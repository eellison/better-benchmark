"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 512, 768], f32), T([4, 512, 1], f32), T([4, 512, 1], f32), T([768], f16), T([768], f16), T([768, 768], f16), T([768, 768], f16), T([768, 768], f16))"

class Repro(torch.nn.Module):
    def forward(self, convert_element_type: "f32[4, 512, 768]", getitem_1: "f32[4, 512, 1]", getitem: "f32[4, 512, 1]", arg0_1: "f16[768]", arg1_1: "f16[768]", arg3_1: "f16[768, 768]", arg5_1: "f16[768, 768]", arg7_1: "f16[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:215 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        sub_tensor: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_tensor: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg0_1);  mul_tensor = arg0_1 = None
        add_tensor_1: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1_1);  mul_tensor_1 = arg1_1 = None
        convert_element_type_default: "f16[4, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:151 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        reshape_default: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, [2048, 768])
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:154 in forward, code: key_states = self.k_proj(hidden_states)
        reshape_default_1: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, [2048, 768])
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:155 in forward, code: value_states = self.v_proj(hidden_states)
        reshape_default_2: "f16[2048, 768]" = torch.ops.aten.reshape.default(convert_element_type_default, [2048, 768]);  convert_element_type_default = None
        permute_default_2: "f16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)


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
