"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: 851ae8199532
Shape hash: 3bd04781
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
    def forward(self, arg0_1: "bf16[128, 16384]", arg1_1: "bf16[128, 16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        mul: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  mul = None
        convert_element_type_1: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 3.0)
        mul_2: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        mul_3: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_3);  mul_3 = None
        add_1: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_4: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type, add_1);  convert_element_type = add_1 = None
        convert_element_type_2: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_1, sub);  mul_1 = sub = None
        mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, 0.7978845608028654);  mul_6 = None
        convert_element_type_3: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16)
        mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_7, 0.044715);  mul_7 = None
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2.0);  convert_element_type_1 = None
        mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_10: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        convert_element_type_4: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        add_2: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_3, convert_element_type_4);  convert_element_type_3 = convert_element_type_4 = None
        mul_11: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.5);  convert_element_type_2 = None
        add_3: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_2, mul_11);  add_2 = mul_11 = None
        view_2: "bf16[128, 16384]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        permute: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[16384]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type_5: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_6: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        return (view_2, permute, convert_element_type_6)



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
