"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 3116aaaaedd1
Shape hash: f749e533
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
    def forward(self, arg0_1: "bf16[4096, 1024]", arg1_1: "b8[32, 128, 1024]", arg2_1: "bf16[4096, 1024]", arg3_1: "bf16[4096, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view_1: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mul_2: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        convert_element_type_2: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        pow_1: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 3.0)
        mul_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_1, mul_3);  view_1 = mul_3 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_2, add_1)
        mul_6: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, mul_5);  mul_5 = None
        view_2: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        mul_7: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, view_2);  mul_1 = view_2 = None
        convert_element_type_3: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        view_3: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_3, [1, 0])
        mul_8: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, mul_2);  mul_2 = None
        mul_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, add_1);  mul_7 = add_1 = None
        convert_element_type_4: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_10);  mul_10 = None
        mul_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_8, sub);  mul_8 = sub = None
        mul_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_11, 0.7978845608028654);  mul_11 = None
        convert_element_type_5: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16)
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, 0.044715);  mul_12 = None
        pow_2: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2.0);  convert_element_type_2 = None
        mul_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        convert_element_type_6: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        add_2: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_5, convert_element_type_6);  convert_element_type_5 = convert_element_type_6 = None
        mul_16: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.5);  convert_element_type_4 = None
        add_3: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_2, mul_16);  add_2 = mul_16 = None
        view_4: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_3, _shape_param_4);  add_3 = _shape_param_4 = None
        permute_1: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        return (view_3, permute, view_4, permute_1)



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
