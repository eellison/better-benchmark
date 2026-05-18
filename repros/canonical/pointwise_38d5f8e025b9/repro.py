"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: 38d5f8e025b9
Shape hash: 9a53abb0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_88: "f32[1, 1024, 1]", convert_element_type_66: "f32[1, 1024, 1024]", getitem_89: "f32[1, 1024, 1]", arg179_1: "bf16[1024]", arg180_1: "bf16[1024]", arg181_1: "bf16[1024, 1024]", arg183_1: "bf16[1024, 1024]", arg185_1: "bf16[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_default: "f32[1, 1024, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_66, getitem_89);  convert_element_type_66 = getitem_89 = None
        mul_tensor: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg179_1);  mul_tensor = arg179_1 = None
        add_tensor_1: "f32[1, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg180_1);  mul_tensor_1 = arg180_1 = None
        convert_element_type_default: "bf16[1, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        reshape_default: "bf16[1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_0);  _shape_param_0 = None
        permute_default: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        reshape_default_1: "bf16[1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_1);  _shape_param_1 = None
        permute_default_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        reshape_default_2: "bf16[1024, 1024]" = torch.ops.aten.reshape.default(convert_element_type_default, _shape_param_2);  convert_element_type_default = _shape_param_2 = None
        permute_default_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_2, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([1, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.bfloat16, device='cuda'),
    [1024, 1024],  # _shape_param_0
    [1024, 1024],  # _shape_param_1
    [1024, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
