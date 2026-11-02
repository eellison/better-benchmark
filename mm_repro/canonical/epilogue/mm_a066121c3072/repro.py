"""Curated MM repro. Pattern hash: a066121c3072."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096, 16384]", arg1_1: "bf16[4096, 16384]", arg2_1: "bf16[4096, 16384]", arg3_1: "bf16[4096, 16384]", arg4_1: "bf16[4096, 16384]", arg5_1: "bf16[4096, 16384]", arg6_1: "bf16[4096, 16384]", arg7_1: "bf16[4096, 16384]", arg8_1: "bf16[4096, 16384]", arg9_1: "bf16[4096, 16384]", arg10_1: "bf16[4096, 16384]", arg11_1: "bf16[4096, 4096]", arg12_1: "bf16[4096, 16384]", arg13_1: "bf16[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        sum_1: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0], True, dtype = torch.float32);  arg0_1 = None
        view: "f32[16384]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_1: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        sum_2: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0], True, dtype = torch.float32);  arg1_1 = None
        view_1: "f32[16384]" = torch.ops.aten.view.default(sum_2, _shape_param_1);  sum_2 = _shape_param_1 = None
        convert_element_type_2: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_3: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        add: "f32[16384]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        sum_3: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg2_1, [0], True, dtype = torch.float32);  arg2_1 = None
        view_2: "f32[16384]" = torch.ops.aten.view.default(sum_3, _shape_param_2);  sum_3 = _shape_param_2 = None
        convert_element_type_4: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_5: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        add_1: "f32[16384]" = torch.ops.aten.add.Tensor(add, convert_element_type_5);  add = convert_element_type_5 = None
        sum_4: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg3_1, [0], True, dtype = torch.float32);  arg3_1 = None
        view_3: "f32[16384]" = torch.ops.aten.view.default(sum_4, _shape_param_3);  sum_4 = _shape_param_3 = None
        convert_element_type_6: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_7: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        add_2: "f32[16384]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_7);  add_1 = convert_element_type_7 = None
        sum_5: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg4_1, [0], True, dtype = torch.float32);  arg4_1 = None
        view_4: "f32[16384]" = torch.ops.aten.view.default(sum_5, _shape_param_4);  sum_5 = _shape_param_4 = None
        convert_element_type_8: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_9: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        add_3: "f32[16384]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_9);  add_2 = convert_element_type_9 = None
        sum_6: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg5_1, [0], True, dtype = torch.float32);  arg5_1 = None
        view_5: "f32[16384]" = torch.ops.aten.view.default(sum_6, _shape_param_5);  sum_6 = _shape_param_5 = None
        convert_element_type_10: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        convert_element_type_11: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_10, torch.float32);  convert_element_type_10 = None
        add_4: "f32[16384]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_11);  add_3 = convert_element_type_11 = None
        sum_7: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg6_1, [0], True, dtype = torch.float32);  arg6_1 = None
        view_6: "f32[16384]" = torch.ops.aten.view.default(sum_7, _shape_param_6);  sum_7 = _shape_param_6 = None
        convert_element_type_12: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_13: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        add_5: "f32[16384]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_13);  add_4 = convert_element_type_13 = None
        sum_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg7_1, [0], True, dtype = torch.float32);  arg7_1 = None
        view_7: "f32[16384]" = torch.ops.aten.view.default(sum_8, _shape_param_7);  sum_8 = _shape_param_7 = None
        convert_element_type_14: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        convert_element_type_15: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        add_6: "f32[16384]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_15);  add_5 = convert_element_type_15 = None
        sum_9: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg8_1, [0], True, dtype = torch.float32);  arg8_1 = None
        view_8: "f32[16384]" = torch.ops.aten.view.default(sum_9, _shape_param_8);  sum_9 = _shape_param_8 = None
        convert_element_type_16: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        convert_element_type_17: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        add_7: "f32[16384]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_17);  add_6 = convert_element_type_17 = None
        sum_10: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg9_1, [0], True, dtype = torch.float32);  arg9_1 = None
        view_9: "f32[16384]" = torch.ops.aten.view.default(sum_10, _shape_param_9);  sum_10 = _shape_param_9 = None
        convert_element_type_18: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        convert_element_type_19: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        add_8: "f32[16384]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_19);  add_7 = convert_element_type_19 = None
        sum_11: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(arg10_1, [0], True, dtype = torch.float32);  arg10_1 = None
        view_10: "f32[16384]" = torch.ops.aten.view.default(sum_11, _shape_param_10);  sum_11 = _shape_param_10 = None
        convert_element_type_20: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        convert_element_type_21: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        add_9: "f32[16384]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_21);  add_8 = convert_element_type_21 = None
        mm: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg11_1, arg12_1);  arg11_1 = arg12_1 = None
        view_11: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm, _shape_param_11);  mm = _shape_param_11 = None
        convert_element_type_22: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        view_12: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg13_1, _shape_param_12);  arg13_1 = _shape_param_12 = None
        mul: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_12, 0.5)
        mul_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_22, mul);  mul = None
        convert_element_type_23: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32)
        pow_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 3.0)
        mul_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_12, mul_2);  view_12 = mul_2 = None
        mul_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_3);  mul_3 = None
        add_11: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_22, add_11);  convert_element_type_22 = add_11 = None
        convert_element_type_24: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_1, sub);  mul_1 = sub = None
        mul_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_6, 0.7978845608028654);  mul_6 = None
        convert_element_type_25: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16)
        mul_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_7, 0.044715);  mul_7 = None
        pow_2: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2.0);  convert_element_type_23 = None
        mul_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        convert_element_type_26: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        add_12: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_25, convert_element_type_26);  convert_element_type_25 = convert_element_type_26 = None
        mul_11: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5);  convert_element_type_24 = None
        add_13: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_12, mul_11);  add_12 = mul_11 = None
        view_13: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_13, _shape_param_13);  add_13 = _shape_param_13 = None
        permute: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_13, [1, 0])
        sum_12: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_13, [0], True, dtype = torch.float32)
        view_14: "f32[16384]" = torch.ops.aten.view.default(sum_12, _shape_param_14);  sum_12 = _shape_param_14 = None
        convert_element_type_27: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        convert_element_type_28: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        add_14: "f32[16384]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_28);  add_9 = convert_element_type_28 = None
        return (view_13, permute, add_14)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
