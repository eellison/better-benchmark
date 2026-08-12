"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: b6027582bc37
Shape hash: 4e691609
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
    def forward(self, arg0_1: "bf16[250112, 512]", arg1_1: "bf16[4096, 512]", arg2_1: "bf16[4096, 512]", arg3_1: "bf16[4096, 512]", arg4_1: "f32[512]", arg5_1: "b8[32, 128, 512]", arg6_1: "f32[32, 128, 512]", arg7_1: "f32[32, 128, 1]", arg8_1: "f32[32, 128, 512]", arg9_1: "i64[32, 128]", arg10_1: "bf16[4096, 512]", arg11_1: "bf16[4096, 512]", arg12_1: "bf16[4096, 512]", arg13_1: "f32[512]", arg14_1: "b8[32, 128, 512]", arg15_1: "f32[32, 128, 1]", arg16_1: "f32[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        convert_element_type: "f32[250112, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type_1: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_2: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        view_2: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_3: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add, convert_element_type_3);  add = convert_element_type_3 = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg4_1);  arg4_1 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg5_1, arg6_1)
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_2, arg7_1)
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, mul_3);  add_1 = mul_3 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_4, [0, 1], True, dtype = torch.float32);  mul_4 = None
        view_3: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, mul_2)
        mul_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, arg7_1);  mul = None
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True, dtype = torch.float32);  mul_5 = None
        add_2: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg8_1, mul_6);  arg8_1 = mul_6 = None
        pow_1: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg7_1, 3);  arg7_1 = None
        mul_7: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_8: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_7, pow_1);  mul_7 = pow_1 = None
        expand: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_8, _shape_param_4);  mul_8 = _shape_param_4 = None
        div: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand, 512);  expand = None
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_2, 1.0);  mul_2 = None
        mul_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div, mul_9);  div = mul_9 = None
        add_3: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_2, mul_10);  add_2 = mul_10 = None
        convert_element_type_4: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_11: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_12: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_3, mul_11);  add_3 = mul_11 = None
        ge: "b8[32, 128]" = torch.ops.aten.ge.Scalar(arg9_1, 0)
        lt: "b8[32, 128]" = torch.ops.aten.lt.Scalar(arg9_1, 250112)
        bitwise_and: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg9_1, -1)
        bitwise_and_1: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[250112, 512]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate: "f32[250112, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg9_1], mul_12);  mul_12 = None
        add_4: "f32[250112, 512]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate);  convert_element_type = _unsafe_masked_index_put_accumulate = None
        view_4: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg10_1, _shape_param_6);  arg10_1 = _shape_param_6 = None
        convert_element_type_5: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        view_5: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg11_1, _shape_param_7);  arg11_1 = _shape_param_7 = None
        convert_element_type_6: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        add_5: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_5, convert_element_type_6);  convert_element_type_5 = convert_element_type_6 = None
        view_6: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg12_1, _shape_param_8);  arg12_1 = _shape_param_8 = None
        convert_element_type_7: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        add_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_7);  add_5 = convert_element_type_7 = None
        mul_13: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, arg13_1);  arg13_1 = None
        mul_14: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg14_1, arg6_1);  arg6_1 = None
        mul_15: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        mul_16: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_15, arg15_1)
        mul_17: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, mul_16);  add_6 = mul_16 = None
        sum_3: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_17, [0, 1], True, dtype = torch.float32);  mul_17 = None
        view_7: "f32[512]" = torch.ops.aten.view.default(sum_3, _shape_param_9);  sum_3 = _shape_param_9 = None
        mul_18: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_13, mul_15)
        mul_19: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_13, arg15_1);  mul_13 = None
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True, dtype = torch.float32);  mul_18 = None
        add_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg16_1, mul_19);  arg16_1 = mul_19 = None
        pow_3: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg15_1, 3);  arg15_1 = None
        mul_20: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_21: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_20, pow_3);  mul_20 = pow_3 = None
        expand_1: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_21, _shape_param_10);  mul_21 = _shape_param_10 = None
        div_1: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_4: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_15, 1.0);  mul_15 = None
        mul_22: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_23: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_1, mul_22);  div_1 = mul_22 = None
        add_8: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_7, mul_23);  add_7 = mul_23 = None
        convert_element_type_8: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        mul_24: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_25: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_8, mul_24);  add_8 = mul_24 = None
        _unsafe_masked_index_put_accumulate_1: "f32[250112, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg9_1], mul_25);  full = unsqueeze = arg9_1 = mul_25 = None
        add_9: "f32[250112, 512]" = torch.ops.aten.add.Tensor(add_4, _unsafe_masked_index_put_accumulate_1);  add_4 = _unsafe_masked_index_put_accumulate_1 = None
        return (view_3, view_7, add_9)



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
