"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 00516eacb000
Shape hash: f40e439b
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
    def forward(self, arg0_1: "bf16[32768, 128]", arg1_1: "f32[256, 128, 128]", arg2_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[256, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        sum_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1], True, dtype = torch.float32)
        view_1: "f32[128]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  convert_element_type = arg2_1 = None
        sum_2: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul, [0, 1], True, dtype = torch.float32);  mul = None
        view_2: "f32[128]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        convert_element_type_1: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        view_3: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_3);  convert_element_type_1 = _shape_param_3 = None
        permute: "bf16[128, 32768]" = torch.ops.aten.permute.default(view_3, [1, 0])
        sum_3: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[128]" = torch.ops.aten.view.default(sum_3, _shape_param_4);  sum_3 = _shape_param_4 = None
        convert_element_type_2: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_3: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        return (view_1, mul_1, view_2, view_3, permute, convert_element_type_3)



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
