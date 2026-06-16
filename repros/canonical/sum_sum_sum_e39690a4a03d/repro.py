"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: e39690a4a03d
Shape hash: 2394756c
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
    def forward(self, arg0_1: "bf16[32768, 512]", arg1_1: "f32[256, 128, 512]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[32768, 512]", arg4_1: "f32[256, 128, 512]", arg5_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        view: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        view_2: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(add_2, [0, 1], True, dtype = torch.float32)
        view_3: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_2, arg4_1);  arg4_1 = None
        mul_1: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_2, arg5_1);  add_2 = arg5_1 = None
        sum_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul, [0, 1], True, dtype = torch.float32);  mul = None
        view_4: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_4);  sum_2 = _shape_param_4 = None
        convert_element_type_3: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        view_5: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_5);  convert_element_type_3 = _shape_param_5 = None
        permute: "bf16[512, 32768]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_3: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[512]" = torch.ops.aten.view.default(sum_3, _shape_param_6);  sum_3 = _shape_param_6 = None
        convert_element_type_4: "bf16[512]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_5: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (view_3, mul_1, view_4, view_5, permute, convert_element_type_5)



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
