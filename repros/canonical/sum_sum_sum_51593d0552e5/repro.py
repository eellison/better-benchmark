"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 51593d0552e5
Shape hash: fd972c32
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
    def forward(self, arg0_1: "bf16[32768, 128]", arg1_1: "f32[256, 128, 128]", arg2_1: "bf16[32768, 128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "bf16[32768, 128]", arg6_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        sum_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add, [0, 1], True, dtype = torch.float32)
        view_1: "f32[128]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        view_2: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_2, arg3_1)
        add_1: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg5_1, _shape_param_3);  arg5_1 = _shape_param_3 = None
        add_2: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_3, add_1);  view_3 = add_1 = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, add_2);  add_2 = None
        mul_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, arg6_1);  add = arg6_1 = None
        sum_2: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1], True, dtype = torch.float32);  mul_1 = None
        view_4: "f32[128]" = torch.ops.aten.view.default(sum_2, _shape_param_4);  sum_2 = _shape_param_4 = None
        convert_element_type_1: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16)
        view_5: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  convert_element_type_1 = _shape_param_5 = None
        permute: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_3: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[128]" = torch.ops.aten.view.default(sum_3, _shape_param_6);  sum_3 = _shape_param_6 = None
        convert_element_type_2: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        sum_4: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True, dtype = torch.float32)
        view_7: "f32[128]" = torch.ops.aten.view.default(sum_4, _shape_param_7);  sum_4 = _shape_param_7 = None
        mul_3: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_2, view_2);  view_2 = None
        mul_4: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        convert_element_type_4: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        sum_5: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True, dtype = torch.float32);  mul_3 = None
        view_8: "f32[128]" = torch.ops.aten.view.default(sum_5, _shape_param_8);  sum_5 = _shape_param_8 = None
        view_9: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_9);  convert_element_type_4 = _shape_param_9 = None
        permute_1: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_9, [1, 0])
        sum_6: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32)
        view_10: "f32[128]" = torch.ops.aten.view.default(sum_6, _shape_param_10);  sum_6 = _shape_param_10 = None
        convert_element_type_5: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        convert_element_type_6: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (view_1, view_4, view_5, permute, convert_element_type_3, view_7, view_8, view_9, permute_1, convert_element_type_6)



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
