"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 1db528935b36
Shape hash: 3c27f190
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
    def forward(self, arg0_1: "bf16[128, 2560, 7, 7]", arg1_1: "f32[2560]", arg2_1: "f32[2560]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 2560, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[128, 2560, 7, 7]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[128, 2560, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_2: "f32[128, 2560, 7, 7]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32)
        pow_1: "f32[128, 2560, 7, 7]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2);  convert_element_type_2 = None
        sum_1: "f32[128, 2560, 1, 1]" = torch.ops.aten.sum.dim_IntList(pow_1, [2, 3], True);  pow_1 = None
        pow_2: "f32[128, 2560, 1, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        mean: "f32[128, 1, 1, 1]" = torch.ops.aten.mean.dim(pow_2, [1], True)
        add_1: "f32[128, 1, 1, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        div: "f32[128, 2560, 1, 1]" = torch.ops.aten.div.Tensor(pow_2, add_1)
        view: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg1_1, [1, -1, 1, 1]);  arg1_1 = None
        view_1: "f32[1, 2560, 1, 1]" = torch.ops.aten.view.default(arg2_1, [1, -1, 1, 1]);  arg2_1 = None
        mul_3: "f32[128, 2560, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_1, div);  div = None
        addcmul: "f32[128, 2560, 7, 7]" = torch.ops.aten.addcmul.default(view, view_1, mul_3);  view = view_1 = mul_3 = None
        add_2: "f32[128, 2560, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_1, addcmul);  convert_element_type_1 = addcmul = None
        convert_element_type_3: "bf16[128, 2560, 7, 7]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        return (pow_2, add_1, convert_element_type_3)



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
