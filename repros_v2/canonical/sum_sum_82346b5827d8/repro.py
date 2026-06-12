"""
Standalone repro captured via capture_hook.
Label: torchbench_mnasnet1_0_train
Pattern hash: 82346b5827d8
Shape hash: 36943a14
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
    def forward(self, arg0_1: "b8[32, 1280]", arg1_1: "bf16[32, 1280]", arg2_1: "bf16[32, 1280, 7, 7]", arg3_1: "f32[1, 1280, 1, 1]", arg4_1: "f32[1, 1280, 1, 1]", arg5_1: "f32[1280]", arg6_1: "f32[1280]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 1280]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        div: "bf16[32, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul: "bf16[32, 1280]" = torch.ops.aten.mul.Tensor(arg1_1, div);  arg1_1 = div = None
        unsqueeze: "bf16[32, 1280, 1]" = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        unsqueeze_1: "bf16[32, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        expand: "bf16[32, 1280, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        div_1: "bf16[32, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
        sub: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg2_1, arg3_1)
        mul_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        unsqueeze_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1)
        unsqueeze_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_2: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        unsqueeze_4: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_5: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        add: "f32[32, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_5);  mul_2 = unsqueeze_5 = None
        convert_element_type_1: "bf16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        relu: "bf16[32, 1280, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_1);  convert_element_type_1 = None
        le: "b8[32, 1280, 7, 7]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1280, 7, 7]" = torch.ops.aten.where.self(le, full, div_1);  le = div_1 = None
        convert_element_type_2: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze: "f32[1280]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze_6: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_7: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        sum_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sub_1: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_8);  convert_element_type_3 = unsqueeze_8 = None
        mul_3: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1)
        sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3]);  mul_3 = None
        mul_4: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_1, 0.0006377551020408163)
        unsqueeze_9: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_4, 0);  mul_4 = None
        unsqueeze_10: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_5: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.0006377551020408163)
        squeeze_1: "f32[1280]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        mul_6: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_7: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_5, mul_6);  mul_5 = mul_6 = None
        unsqueeze_12: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_7, 0);  mul_7 = None
        unsqueeze_13: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 2);  unsqueeze_12 = None
        unsqueeze_14: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 3);  unsqueeze_13 = None
        mul_8: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_1, arg5_1);  arg5_1 = None
        unsqueeze_15: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_16: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 2);  unsqueeze_15 = None
        unsqueeze_17: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 3);  unsqueeze_16 = None
        mul_9: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_14);  sub_1 = unsqueeze_14 = None
        sub_2: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_9);  convert_element_type_2 = mul_9 = None
        sub_3: "f32[32, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_11);  sub_2 = unsqueeze_11 = None
        mul_10: "f32[32, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_17);  sub_3 = unsqueeze_17 = None
        mul_11: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_4: "bf16[32, 1280, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        return (full, sum_1, mul_11, convert_element_type_4)



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
