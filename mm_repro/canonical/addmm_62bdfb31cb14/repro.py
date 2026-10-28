"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/BlenderbotForConditionalGeneration/full_graph_007.py
Pattern hash: 62bdfb31cb14
Shape hash: d681ef68
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
    def forward(self, arg0_1: "f32[2560]", arg1_1: "f32[2560, 2560]", arg2_1: "bf16[2048, 2560]", arg3_1: "f32[2560]", arg4_1: "f32[2560, 2560]", arg5_1: "f32[16, 128, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = arg2_1 = permute = None
        view: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type_2: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_3: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_4: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        view_2: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_2);  convert_element_type_4 = _shape_param_2 = None
        permute_2: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm_1: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_2, view_2, permute_2);  convert_element_type_2 = view_2 = permute_2 = None
        view_3: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_1, _shape_param_3);  addmm_1 = _shape_param_3 = None
        view_4: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        mul: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_1, 0.334370152488211);  permute_1 = None
        permute_4: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.334370152488211);  permute_4 = None
        expand: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul, _shape_param_5);  mul = _shape_param_5 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_5: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_6);  clone = _shape_param_6 = None
        expand_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_1, _shape_param_7);  mul_1 = _shape_param_7 = None
        clone_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_6: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        permute_5: "bf16[512, 80, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1])
        permute_6: "bf16[512, 128, 80]" = torch.ops.aten.permute.default(view_6, [0, 2, 1])
        return (view_5, view_6, permute_5, permute_6)



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
