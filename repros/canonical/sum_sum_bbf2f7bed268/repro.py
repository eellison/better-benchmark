"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_train
Pattern hash: bbf2f7bed268
Shape hash: d90f6100
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
    def forward(self, arg0_1: "bf16[512, 960, 1, 1]", arg1_1: "bf16[512, 960, 7, 7]", arg2_1: "f32[1, 960, 1, 1]", arg3_1: "f32[1, 960, 1, 1]", arg4_1: "f32[960]", arg5_1: "f32[960]", arg6_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        squeeze: "bf16[512, 960, 1]" = torch.ops.aten.squeeze.dim(arg0_1, 3);  arg0_1 = None
        squeeze_1: "bf16[512, 960]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[491520]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        as_strided_scatter: "bf16[491520]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, _shape_param_1, _shape_param_2, 0);  full = squeeze_1 = _shape_param_1 = _shape_param_2 = None
        as_strided: "bf16[512, 960, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, _shape_param_3, _shape_param_4, 0);  as_strided_scatter = _shape_param_3 = _shape_param_4 = None
        expand: "bf16[512, 960, 7, 7]" = torch.ops.aten.expand.default(as_strided, _shape_param_5);  as_strided = _shape_param_5 = None
        div: "bf16[512, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        convert_element_type: "f32[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(div, torch.float32);  div = None
        sub: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1)
        mul: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg3_1);  sub = None
        unsqueeze: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1)
        unsqueeze_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_2: "f32[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        le: "b8[512, 960, 7, 7]" = torch.ops.aten.le.Scalar(convert_element_type_2, -3)
        lt: "b8[512, 960, 7, 7]" = torch.ops.aten.lt.Scalar(convert_element_type_2, 3)
        div_1: "f32[512, 960, 7, 7]" = torch.ops.aten.div.Tensor(convert_element_type_2, 3);  convert_element_type_2 = None
        add_1: "f32[512, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_1, 0.5);  div_1 = None
        mul_2: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  add_1 = None
        where: "f32[512, 960, 7, 7]" = torch.ops.aten.where.self(lt, mul_2, convert_element_type);  lt = mul_2 = convert_element_type = None
        where_1: "f32[512, 960, 7, 7]" = torch.ops.aten.where.self(le, arg6_1, where);  le = arg6_1 = where = None
        convert_element_type_3: "bf16[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        convert_element_type_4: "f32[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        squeeze_2: "f32[960]" = torch.ops.aten.squeeze.dims(arg2_1, [0, 2, 3]);  arg2_1 = None
        unsqueeze_4: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_2, 0);  squeeze_2 = None
        unsqueeze_5: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[960]" = torch.ops.aten.sum.dim_IntList(convert_element_type_4, [0, 2, 3])
        convert_element_type_5: "f32[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub_1: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_6);  convert_element_type_5 = unsqueeze_6 = None
        mul_3: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_1)
        sum_2: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3]);  mul_3 = None
        mul_4: "f32[960]" = torch.ops.aten.mul.Tensor(sum_1, 3.985969387755102e-05)
        unsqueeze_7: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_8: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_5: "f32[960]" = torch.ops.aten.mul.Tensor(sum_2, 3.985969387755102e-05)
        squeeze_3: "f32[960]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        mul_6: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_7: "f32[960]" = torch.ops.aten.mul.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze_10: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_11: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_8: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_3, arg4_1);  arg4_1 = None
        unsqueeze_13: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_14: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_9: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_4, mul_9);  convert_element_type_4 = mul_9 = None
        sub_3: "f32[512, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_10: "f32[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_11: "f32[960]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_3);  sum_2 = squeeze_3 = None
        convert_element_type_6: "bf16[512, 960, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        return (sum_1, mul_11, convert_element_type_6)



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
