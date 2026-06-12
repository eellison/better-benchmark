"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 3280cef0c732
Shape hash: d01c3ddd
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
    def forward(self, arg0_1: "f32[512, 672, 1, 1]", arg1_1: "bf16[512, 672, 14, 14]", arg2_1: "bf16[512, 672, 1, 1]", arg3_1: "bf16[512, 336, 14, 14]", arg4_1: "f32[1, 336, 1, 1]", arg5_1: "f32[1, 336, 1, 1]", arg6_1: "f32[336]", arg7_1: "f32[336]", arg8_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        add: "f32[512, 672, 1, 1]" = torch.ops.aten.add.Tensor(arg0_1, 3);  arg0_1 = None
        clamp_min: "f32[512, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[512, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        div: "f32[512, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
        convert_element_type: "bf16[512, 672, 1, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[512, 672, 14, 14]" = torch.ops.aten.mul.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        expand: "bf16[512, 672, 14, 14]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        div_1: "bf16[512, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
        add_1: "bf16[512, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul, div_1);  mul = div_1 = None
        slice_1: "bf16[512, 336, 14, 14]" = torch.ops.aten.slice.Tensor(add_1, 1, 336, 672)
        sub: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(arg3_1, arg4_1)
        mul_1: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub, arg5_1);  sub = None
        unsqueeze: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1)
        unsqueeze_1: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_2: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_1);  mul_1 = unsqueeze_1 = None
        unsqueeze_2: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_3: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_2: "f32[512, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_3);  mul_2 = unsqueeze_3 = None
        convert_element_type_1: "bf16[512, 336, 14, 14]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        relu: "bf16[512, 336, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        le: "b8[512, 336, 14, 14]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where: "bf16[512, 336, 14, 14]" = torch.ops.aten.where.self(le, arg8_1, slice_1);  le = arg8_1 = slice_1 = None
        convert_element_type_2: "f32[512, 336, 14, 14]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[336]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        unsqueeze_4: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[336]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[512, 336, 14, 14]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub_1: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_6);  convert_element_type_3 = unsqueeze_6 = None
        mul_3: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1)
        sum_2: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3]);  mul_3 = None
        mul_4: "f32[336]" = torch.ops.aten.mul.Tensor(sum_1, 9.964923469387754e-06)
        unsqueeze_7: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_8: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_5: "f32[336]" = torch.ops.aten.mul.Tensor(sum_2, 9.964923469387754e-06)
        squeeze_1: "f32[336]" = torch.ops.aten.squeeze.dims(arg5_1, [0, 2, 3]);  arg5_1 = None
        mul_6: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_7: "f32[336]" = torch.ops.aten.mul.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze_10: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_11: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_8: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_1, arg6_1);  arg6_1 = None
        unsqueeze_13: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_14: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_9: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_9);  convert_element_type_2 = mul_9 = None
        sub_3: "f32[512, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_10: "f32[512, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_11: "f32[336]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_4: "bf16[512, 336, 14, 14]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        return (add_1, sum_1, mul_11, convert_element_type_4)



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
