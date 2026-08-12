"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 2e7998fcb884
Shape hash: b3c0b271
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "f32[8, 512, 4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "f32[4096]", arg5_1: "f32[8, 512, 4096]", arg6_1: "f32[8, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        view_2: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        mul: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_2, arg4_1);  arg4_1 = None
        mul_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul, 4096)
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg5_1, sum_2);  arg5_1 = sum_2 = None
        sub: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg6_1, sub_1);  arg6_1 = sub_1 = None
        convert_element_type_3: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view_3: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_3, [1, 0])
        return (add_2, mul_4, view_3, permute)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
