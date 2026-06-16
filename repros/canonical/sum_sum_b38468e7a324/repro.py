"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v2_train
Pattern hash: b38468e7a324
Shape hash: 114a3974
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
    def forward(self, arg0_1: "b8[96, 1280]", arg1_1: "bf16[96, 1280]", arg2_1: "bf16[96, 1280, 7, 7]", arg3_1: "f32[1, 1280, 1, 1]", arg4_1: "f32[1, 1280, 1, 1]", arg5_1: "f32[1280]", arg6_1: "f32[1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[96, 1280]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mul: "bf16[96, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.25);  convert_element_type = None
        mul_1: "bf16[96, 1280]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        view: "bf16[96, 1280, 1, 1]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        expand: "bf16[96, 1280, 7, 7]" = torch.ops.aten.expand.default(view, _shape_param_1);  view = _shape_param_1 = None
        div: "bf16[96, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[96, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul_2: "f32[96, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        unsqueeze: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1)
        unsqueeze_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_3: "f32[96, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        unsqueeze_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[96, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_3, unsqueeze_3);  mul_3 = unsqueeze_3 = None
        convert_element_type_1: "bf16[96, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        le: "b8[96, 1280, 7, 7]" = torch.ops.aten.le.Scalar(convert_element_type_1, 0.0)
        ge: "b8[96, 1280, 7, 7]" = torch.ops.aten.ge.Scalar(convert_element_type_1, 6.0);  convert_element_type_1 = None
        bitwise_or: "b8[96, 1280, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[96, 1280, 7, 7]" = torch.ops.aten.where.self(bitwise_or, full, div);  bitwise_or = div = None
        convert_element_type_2: "f32[96, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1280]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_4: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_5: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        unsqueeze_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        sum_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[96, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[96, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_6);  convert_element_type_3 = unsqueeze_6 = None
        mul_4: "f32[96, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1)
        sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 2, 3]);  mul_4 = None
        mul_5: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_1, 0.00021258503401360543)
        unsqueeze_7: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_8: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_6: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.00021258503401360543)
        squeeze_1: "f32[1280]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        mul_7: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_8: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        unsqueeze_10: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_11: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_9: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_1, arg5_1);  arg5_1 = None
        unsqueeze_13: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_14: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_10: "f32[96, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_2: "f32[96, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_10);  convert_element_type_2 = mul_10 = None
        sub_3: "f32[96, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_9);  sub_2 = unsqueeze_9 = None
        mul_11: "f32[96, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_15);  sub_3 = unsqueeze_15 = None
        mul_12: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_4: "bf16[96, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        return (full, sum_1, mul_12, convert_element_type_4)



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
