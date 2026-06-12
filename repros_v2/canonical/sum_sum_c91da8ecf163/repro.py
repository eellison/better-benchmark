"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: c91da8ecf163
Shape hash: eeebf109
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
    def forward(self, arg0_1: "bf16[128, 32, 112, 112]", arg1_1: "f32[1, 32, 1, 1]", arg2_1: "f32[1, 32, 1, 1]", arg3_1: "f32[32]", arg4_1: "f32[32]", arg5_1: "bf16[128, 32, 112, 112]", arg6_1: "bf16[128, 32, 1, 1]"):
        # No stacktrace found for following nodes
        sub: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, arg2_1);  sub = arg2_1 = None
        unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        convert_element_type_1: "f32[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        neg: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(convert_element_type_1)
        exp: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg);  neg = None
        add_1: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(convert_element_type_1, add_1);  add_1 = None
        convert_element_type_2: "bf16[128, 32, 112, 112]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        mul_2: "bf16[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(arg5_1, convert_element_type_2);  arg5_1 = convert_element_type_2 = None
        sigmoid: "bf16[128, 32, 1, 1]" = torch.ops.aten.sigmoid.default(arg6_1);  arg6_1 = None
        sum_1: "f32[128, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2, 3], True, dtype = torch.float32);  mul_2 = None
        convert_element_type_3: "bf16[128, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_4: "f32[128, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        convert_element_type_5: "f32[128, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32)
        sub_1: "f32[128, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_5)
        mul_3: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_5, sub_1);  convert_element_type_5 = sub_1 = None
        mul_4: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_4, mul_3);  convert_element_type_4 = mul_3 = None
        convert_element_type_6: "bf16[128, 32, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        sum_2: "bf16[32]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 2, 3])
        convert_element_type_7: "f32[32]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (convert_element_type_1, sigmoid, convert_element_type_6, convert_element_type_7)



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
