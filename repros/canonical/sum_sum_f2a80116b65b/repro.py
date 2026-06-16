"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: f2a80116b65b
Shape hash: 6b37439d
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
    def forward(self, arg0_1: "f32[512, 72, 1, 1]", arg1_1: "bf16[512, 72, 28, 28]", arg2_1: "bf16[512, 72, 1, 1]", arg3_1: "f32[1, 72, 1, 1]", arg4_1: "bf16[512, 72, 28, 28]", arg5_1: "f32[1, 72, 1, 1]", arg6_1: "f32[72]", _shape_param_0):
        # No stacktrace found for following nodes
        add: "f32[512, 72, 1, 1]" = torch.ops.aten.add.Tensor(arg0_1, 3);  arg0_1 = None
        clamp_min: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add, 0);  add = None
        clamp_max: "f32[512, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        div: "f32[512, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
        convert_element_type: "bf16[512, 72, 1, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul: "bf16[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        expand: "bf16[512, 72, 28, 28]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        div_1: "bf16[512, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand, 784);  expand = None
        add_1: "bf16[512, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul, div_1);  mul = div_1 = None
        convert_element_type_1: "f32[512, 72, 28, 28]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        squeeze: "f32[72]" = torch.ops.aten.squeeze.dims(arg3_1, [0, 2, 3]);  arg3_1 = None
        unsqueeze: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        sum_1: "f32[72]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [0, 2, 3])
        convert_element_type_2: "f32[512, 72, 28, 28]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        sub: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_2);  convert_element_type_2 = unsqueeze_2 = None
        mul_1: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub)
        sum_2: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 2, 3]);  mul_1 = None
        mul_2: "f32[72]" = torch.ops.aten.mul.Tensor(sum_1, 2.4912308673469386e-06)
        unsqueeze_3: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_2, 0);  mul_2 = None
        unsqueeze_4: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_3: "f32[72]" = torch.ops.aten.mul.Tensor(sum_2, 2.4912308673469386e-06)
        squeeze_1: "f32[72]" = torch.ops.aten.squeeze.dims(arg5_1, [0, 2, 3]);  arg5_1 = None
        mul_4: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_5: "f32[72]" = torch.ops.aten.mul.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        unsqueeze_6: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_7: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_6: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_1, arg6_1);  arg6_1 = None
        unsqueeze_9: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_6, 0);  mul_6 = None
        unsqueeze_10: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_7: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_8);  sub = unsqueeze_8 = None
        sub_1: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_7);  convert_element_type_1 = mul_7 = None
        sub_2: "f32[512, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_1, unsqueeze_5);  sub_1 = unsqueeze_5 = None
        mul_8: "f32[512, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_11);  sub_2 = unsqueeze_11 = None
        mul_9: "f32[72]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_3: "bf16[512, 72, 28, 28]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        return (sum_1, mul_9, convert_element_type_3)



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
