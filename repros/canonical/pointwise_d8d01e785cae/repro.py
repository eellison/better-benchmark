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

_shapes_config = "(T([2048, 768], f32), T([2048, 1], f32), T([2048, 1], f32), T([768], f16), T([768], f16), T([3072, 768], f16))"

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_14: "f32[2048, 768]", getitem_12: "f32[2048, 1]", getitem_11: "f32[2048, 1]", arg12_1: "f16[768]", arg13_1: "f16[768]", arg14_1: "f16[3072, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:239 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        sub_tensor: "f32[2048, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_12);  convert_element_type_14 = getitem_12 = None
        add_tensor: "f32[2048, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_default: "f32[2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[2048, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg12_1);  mul_tensor = arg12_1 = None
        add_tensor_1: "f32[2048, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg13_1);  mul_tensor_1 = arg13_1 = None
        convert_element_type_default: "f16[2048, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:241 in forward, code: hidden_states = self.fc1(hidden_states)
        permute_default: "f16[768, 3072]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        return (convert_element_type_default, permute_default)


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
