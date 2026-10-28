"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/lennard_jones/full_graph_001.py
Pattern hash: 9f396db49822
Shape hash: adf4b59c
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
    def forward(self, arg0_1: "bf16[128, 16]", arg1_1: "bf16[16, 16]", arg2_1: "bf16[128, 16]", arg3_1: "bf16[16, 16]", arg4_1: "bf16[128, 16]", arg5_1: "bf16[128, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[128, 16]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_1: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1);  convert_element_type_1 = None
        sub: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul);  mul = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, sub);  convert_element_type = sub = None
        convert_element_type_2: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        mm_1: "bf16[128, 16]" = torch.ops.aten.mm.default(convert_element_type_2, arg3_1);  arg3_1 = None
        permute: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0])
        sum_1: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0], True, dtype = torch.float32);  convert_element_type_2 = None
        view: "f32[16]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_3: "bf16[16]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_6: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_2: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_6, convert_element_type_6);  convert_element_type_6 = None
        sub_1: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul_2);  mul_2 = None
        mul_3: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_1);  convert_element_type_5 = sub_1 = None
        convert_element_type_7: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute_1: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0])
        mm_2: "bf16[16, 1]" = torch.ops.aten.mm.default(permute_1, arg5_1);  permute_1 = arg5_1 = None
        sum_2: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0], True, dtype = torch.float32);  convert_element_type_7 = None
        view_1: "f32[16]" = torch.ops.aten.view.default(sum_2, _shape_param_1);  sum_2 = _shape_param_1 = None
        convert_element_type_8: "bf16[16]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_9: "f32[16, 1]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_10: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        return (permute, convert_element_type_4, convert_element_type_9, convert_element_type_10)



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
