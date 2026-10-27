"""
Standalone repro captured via capture_hook.
Label: mobilenet_v3_large
Pattern hash: 639ec5b7df3a
Shape hash: e8318449
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
    def forward(self, arg0_1: "bf16[32, 1000]", arg1_1: "bf16[1000, 1280]", arg2_1: "b8[32, 1280]", arg3_1: "bf16[32, 1280]", arg4_1: "bf16[1280, 960]", arg5_1: "bf16[32, 960]", arg6_1: "bf16[32, 960, 7, 7]", arg7_1: "f32[1, 960, 1, 1]", arg8_1: "f32[1, 960, 1, 1]", arg9_1: "f32[960]", arg10_1: "f32[960]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        mm: "bf16[32, 1280]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(mm, div);  mm = div = None
        convert_element_type_1: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_2: "f32[32, 1280]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        le: "b8[32, 1280]" = torch.ops.aten.le.Scalar(convert_element_type_2, -3)
        lt: "b8[32, 1280]" = torch.ops.aten.lt.Scalar(convert_element_type_2, 3)
        div_1: "f32[32, 1280]" = torch.ops.aten.div.Tensor(convert_element_type_2, 3);  convert_element_type_2 = None
        add: "f32[32, 1280]" = torch.ops.aten.add.Tensor(div_1, 0.5);  div_1 = None
        mul_1: "f32[32, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_1, add);  add = None
        where: "f32[32, 1280]" = torch.ops.aten.where.self(lt, mul_1, convert_element_type_1);  lt = mul_1 = convert_element_type_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1280]" = torch.ops.aten.where.self(le, full, where);  le = where = None
        convert_element_type_3: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        mm_1: "bf16[32, 960]" = torch.ops.aten.mm.default(convert_element_type_3, arg4_1);  arg4_1 = None
        permute: "bf16[1280, 32]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0])
        mm_2: "bf16[1280, 960]" = torch.ops.aten.mm.default(permute, arg5_1);  permute = arg5_1 = None
        sum_1: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0], True, dtype = torch.float32);  convert_element_type_3 = None
        view: "f32[1280]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_4: "bf16[1280]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_5: "f32[1280, 960]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_6: "f32[1280]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        view_1: "bf16[32, 960, 1, 1]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        expand: "bf16[32, 960, 7, 7]" = torch.ops.aten.expand.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        div_2: "bf16[32, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type_7: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div_2, torch.float32);  div_2 = None
        sub: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg6_1, arg7_1)
        mul_2: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg8_1);  sub = None
        unsqueeze: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1)
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_3: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        unsqueeze_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_1: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_3);  mul_3 = unsqueeze_3 = None
        convert_element_type_8: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_9: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        le_1: "b8[32, 960, 7, 7]" = torch.ops.aten.le.Scalar(convert_element_type_9, -3)
        lt_1: "b8[32, 960, 7, 7]" = torch.ops.aten.lt.Scalar(convert_element_type_9, 3)
        div_3: "f32[32, 960, 7, 7]" = torch.ops.aten.div.Tensor(convert_element_type_9, 3);  convert_element_type_9 = None
        add_2: "f32[32, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_3, 0.5);  div_3 = None
        mul_4: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_7, add_2);  add_2 = None
        where_2: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(lt_1, mul_4, convert_element_type_7);  lt_1 = mul_4 = convert_element_type_7 = None
        where_3: "f32[32, 960, 7, 7]" = torch.ops.aten.where.self(le_1, full, where_2);  le_1 = where_2 = None
        convert_element_type_10: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        convert_element_type_11: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        squeeze: "f32[960]" = torch.ops.aten.squeeze.dims(arg7_1, [0, 2, 3]);  arg7_1 = None
        unsqueeze_4: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_2: "f32[960]" = torch.ops.aten.sum.dim_IntList(convert_element_type_11, [0, 2, 3])
        convert_element_type_12: "f32[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        sub_1: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_12, unsqueeze_6);  convert_element_type_12 = unsqueeze_6 = None
        mul_5: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_11, sub_1)
        sum_3: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 2, 3]);  mul_5 = None
        mul_6: "f32[960]" = torch.ops.aten.mul.Tensor(sum_2, 0.0006377551020408163)
        unsqueeze_7: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_8: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_7: "f32[960]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006377551020408163)
        squeeze_1: "f32[960]" = torch.ops.aten.squeeze.dims(arg8_1, [0, 2, 3]);  arg8_1 = None
        mul_8: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_9: "f32[960]" = torch.ops.aten.mul.Tensor(mul_7, mul_8);  mul_7 = mul_8 = None
        unsqueeze_10: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_11: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_10: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_1, arg9_1);  arg9_1 = None
        unsqueeze_13: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_10, 0);  mul_10 = None
        unsqueeze_14: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_11: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_11, mul_11);  convert_element_type_11 = mul_11 = None
        sub_3: "f32[32, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_12: "f32[32, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_13: "f32[960]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_1);  sum_3 = squeeze_1 = None
        convert_element_type_13: "bf16[32, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        return (full, convert_element_type_5, convert_element_type_6, sum_2, mul_13, convert_element_type_13)



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
