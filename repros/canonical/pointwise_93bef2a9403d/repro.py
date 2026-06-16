"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 93bef2a9403d
Shape hash: 6498d204
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
    def forward(self, arg0_1: "bf16[128, 1536, 1, 1]", arg1_1: "bf16[128, 1536, 12, 12]", arg2_1: "f32[]", arg3_1: "bf16[128, 1536, 12, 12]"):
        # No stacktrace found for following nodes
        sigmoid: "bf16[128, 1536, 1, 1]" = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
        mul: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(arg1_1, sigmoid);  arg1_1 = sigmoid = None
        mul_1: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
        mul_2: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        mul_3: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_2, 0.2);  mul_2 = None
        add: "bf16[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        convert_element_type: "f32[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        mul_4: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_5: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[128, 1536, 12, 12]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[128, 1536, 12, 12]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = add_1 = None
        convert_element_type_1: "bf16[128, 1536, 12, 12]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        mul_7: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.7015043497085571);  convert_element_type_1 = None
        mul_8: "bf16[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_7, 0.8980265101338745);  mul_7 = None
        avg_pool2d: "bf16[128, 1536, 6, 6]" = torch.ops.aten.avg_pool2d.default(mul_8, [2, 2], [2, 2], [0, 0], True, False)
        return (add, mul_8, avg_pool2d)



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
