"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: 292e665668b6
Shape hash: 4cb33acf
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
    def forward(self, arg0_1: "bf16[128, 32, 112, 112]", arg1_1: "bf16[128, 32, 1, 1]", arg2_1: "bf16[128, 32, 1, 1]", arg3_1: "f32[128, 32, 112, 112]", arg4_1: "f32[1, 32, 1, 1]", arg5_1: "bf16[128, 32, 112, 112]", arg6_1: "f32[1, 32, 1, 1]", arg7_1: "f32[32]", _shape_param_0):
        # No stacktrace found for following nodes
        mul: "bf16[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        expand: "bf16[128, 32, 112, 112]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        div: "bf16[128, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand, 12544);  expand = None
        add: "bf16[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul, div);  mul = div = None
        convert_element_type: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        sigmoid: "f32[128, 32, 112, 112]" = torch.ops.aten.sigmoid.default(arg3_1)
        mul_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type, sigmoid);  convert_element_type = None
        sub: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(arg3_1, sub);  arg3_1 = sub = None
        add_1: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, 1);  mul_2 = None
        mul_3: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, add_1);  mul_1 = add_1 = None
        convert_element_type_1: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        convert_element_type_2: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(arg4_1, [0, 2, 3]);  arg4_1 = None
        unsqueeze: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        sum_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 2, 3])
        convert_element_type_3: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        sub_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_2);  convert_element_type_3 = unsqueeze_2 = None
        mul_4: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub_1)
        sum_2: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 2, 3]);  mul_4 = None
        mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(sum_1, 6.228077168367346e-07)
        unsqueeze_3: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_4: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        mul_6: "f32[32]" = torch.ops.aten.mul.Tensor(sum_2, 6.228077168367346e-07)
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(arg6_1, [0, 2, 3]);  arg6_1 = None
        mul_7: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        unsqueeze_6: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_7: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, arg7_1);  arg7_1 = None
        unsqueeze_9: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_9, 0);  mul_9 = None
        unsqueeze_10: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 2);  unsqueeze_9 = None
        unsqueeze_11: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        mul_10: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_8);  sub_1 = unsqueeze_8 = None
        sub_2: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_10);  convert_element_type_2 = mul_10 = None
        sub_3: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_2, unsqueeze_5);  sub_2 = unsqueeze_5 = None
        mul_11: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_11);  sub_3 = unsqueeze_11 = None
        mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_1);  sum_2 = squeeze_1 = None
        convert_element_type_4: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        return (sum_1, mul_12, convert_element_type_4)



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
