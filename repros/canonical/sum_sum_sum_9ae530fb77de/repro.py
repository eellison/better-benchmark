"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 9ae530fb77de
Shape hash: fd33c7c3
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
    def forward(self, arg0_1: "bf16[128, 1536, 6, 6]", arg1_1: "bf16[128, 1536, 1, 1]", arg2_1: "bf16[128, 1536, 6, 6]", arg3_1: "f32[]"):
        # No stacktrace found for following nodes
        mul: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(arg0_1, 0.2);  arg0_1 = None
        sigmoid: "bf16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        mul_1: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(arg2_1, sigmoid)
        mul_2: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_1, 2.0);  mul_1 = None
        mul_3: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul, mul_2);  mul_2 = None
        mul_4: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(mul_3, dtype = torch.float32);  mul_3 = None
        mul_5: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_4, 2.0);  mul_4 = None
        mul_6: "bf16[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_5, arg2_1);  arg2_1 = None
        sum_2: "f32[128, 1536, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2, 3], True, dtype = torch.float32);  mul_6 = None
        convert_element_type: "bf16[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_2, torch.bfloat16);  sum_2 = None
        convert_element_type_1: "f32[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32)
        sub: "f32[128, 1536, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_2)
        mul_7: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, sub);  convert_element_type_2 = sub = None
        mul_8: "f32[128, 1536, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_7);  convert_element_type_1 = mul_7 = None
        convert_element_type_3: "bf16[128, 1536, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        sum_3: "bf16[1536]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[1536]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        return (sigmoid, sum_1, mul_5, convert_element_type_3, convert_element_type_4)



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
