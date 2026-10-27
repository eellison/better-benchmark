"""
Standalone repro captured via capture_hook.
Label: vgg16
Pattern hash: 7bba7475e9a9
Shape hash: 619a81ab
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
    def forward(self, arg0_1: "bf16[64, 1000]", arg1_1: "bf16[1000, 4096]", arg2_1: "b8[64, 4096]", arg3_1: "b8[64, 4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[64, 4096]", arg6_1: "b8[64, 4096]", arg7_1: "b8[64, 4096]", arg8_1: "bf16[4096, 25088]", arg9_1: "bf16[64, 25088]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[64, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[64, 4096]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_1: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(mm, mul);  mm = mul = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[64, 4096]" = torch.ops.aten.where.self(arg3_1, full, mul_1);  arg3_1 = mul_1 = None
        mm_1: "bf16[64, 4096]" = torch.ops.aten.mm.default(where, arg4_1);  arg4_1 = None
        permute: "bf16[4096, 64]" = torch.ops.aten.permute.default(where, [1, 0])
        mm_2: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = arg5_1 = None
        sum_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True, dtype = torch.float32);  where = None
        view: "f32[4096]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_1: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_2: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_3: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        convert_element_type_4: "bf16[64, 4096]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        mul_2: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 2.0);  convert_element_type_4 = None
        mul_3: "bf16[64, 4096]" = torch.ops.aten.mul.Tensor(mm_1, mul_2);  mm_1 = mul_2 = None
        where_1: "bf16[64, 4096]" = torch.ops.aten.where.self(arg7_1, full, mul_3);  arg7_1 = mul_3 = None
        mm_3: "bf16[64, 25088]" = torch.ops.aten.mm.default(where_1, arg8_1);  arg8_1 = None
        permute_1: "bf16[4096, 64]" = torch.ops.aten.permute.default(where_1, [1, 0])
        mm_4: "bf16[4096, 25088]" = torch.ops.aten.mm.default(permute_1, arg9_1);  permute_1 = arg9_1 = None
        sum_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True, dtype = torch.float32);  where_1 = None
        view_1: "f32[4096]" = torch.ops.aten.view.default(sum_2, _shape_param_1);  sum_2 = _shape_param_1 = None
        convert_element_type_5: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_6: "f32[4096, 25088]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_7: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        view_2: "bf16[64, 512, 7, 7]" = torch.ops.aten.view.default(mm_3, _shape_param_2);  mm_3 = _shape_param_2 = None
        return (full, convert_element_type_2, convert_element_type_3, convert_element_type_6, convert_element_type_7, view_2)



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
