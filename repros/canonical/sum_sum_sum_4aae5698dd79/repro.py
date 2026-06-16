"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 4aae5698dd79
Shape hash: 2f93d7aa
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
    def forward(self, arg0_1: "bf16[32, 6, 128, 128]", arg1_1: "bf16[32, 6, 128, 128]", arg2_1: "bf16[32, 6, 128, 128]", arg3_1: "bf16[32, 6, 128, 128]", arg4_1: "bf16[32, 6, 128, 128]", arg5_1: "bf16[32, 6, 128, 128]", arg6_1: "bf16[32, 6, 128, 128]", arg7_1: "bf16[192, 128, 128]", arg8_1: "b8[32, 6, 128, 128]", arg9_1: "i64[1, 1, 128]", arg10_1: "f32[]", arg11_1: "bf16[192, 128, 128]", arg12_1: "f32[128, 128, 6]", arg13_1: "f32[32, 6, 128, 1]", arg14_1: "f32[32, 6, 128, 1]", arg15_1: "i64[128, 128]", arg16_1: "bf16[32, 6, 128, 128]", arg17_1: "bf16[32, 6, 128, 128]", arg18_1: "bf16[32, 6, 128, 128]", arg19_1: "bf16[32, 6, 128, 128]", arg20_1: "bf16[32, 6, 128, 128]", arg21_1: "bf16[32, 6, 128, 128]", arg22_1: "bf16[32, 6, 128, 128]", arg23_1: "bf16[192, 128, 128]", arg24_1: "b8[32, 6, 128, 128]", arg25_1: "bf16[192, 128, 128]", arg26_1: "f32[128, 128, 6]", arg27_1: "f32[32, 6, 128, 1]", arg28_1: "f32[32, 6, 128, 1]", arg29_1: "i64[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        convert_element_type_2: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        convert_element_type_3: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_3: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_4);  add_2 = convert_element_type_4 = None
        convert_element_type_5: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        add_4: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_5);  add_3 = convert_element_type_5 = None
        convert_element_type_6: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        add_5: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_6);  add_4 = convert_element_type_6 = None
        view: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg7_1, _shape_param_0);  arg7_1 = _shape_param_0 = None
        convert_element_type_7: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        mul: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_1: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_8: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        unsqueeze: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, 3)
        unsqueeze_1: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg9_1, 2);  arg9_1 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_1, unsqueeze);  unsqueeze_1 = unsqueeze = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, _shape_param_1);  le = _shape_param_1 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, arg10_1, full);  expand = arg10_1 = full = None
        view_1: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg11_1, _shape_param_2);  arg11_1 = _shape_param_2 = None
        permute: "f32[6, 128, 128]" = torch.ops.aten.permute.default(arg12_1, [2, 0, 1]);  arg12_1 = None
        unsqueeze_2: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        add_6: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_2, where);  unsqueeze_2 = where = None
        add_7: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_1, add_6);  view_1 = add_6 = None
        convert_element_type_9: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        convert_element_type_10: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        sub: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, arg13_1);  convert_element_type_10 = arg13_1 = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg14_1);  exp = arg14_1 = None
        mul_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_8, div);  convert_element_type_8 = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [-1], True)
        neg: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_11: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_2: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_3);  convert_element_type_11 = _shape_param_3 = None
        view_3: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        view_4: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_3, _shape_param_5);  _shape_param_5 = None
        convert_element_type_12: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        add_8: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_12);  add_5 = convert_element_type_12 = None
        sum_2: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_8, [0], True, dtype = torch.float32);  add_8 = None
        squeeze: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_2, 0);  sum_2 = None
        permute_1: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        full_1: "b8[128, 128, 1]" = torch.ops.aten.full.default(_shape_param_6, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        full_2: "f32[32, 6]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        _unsafe_masked_index_put_accumulate: "f32[32, 6]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, full_1, [arg15_1], permute_1);  arg15_1 = permute_1 = None
        convert_element_type_13: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        convert_element_type_14: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        add_9: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_13, convert_element_type_14);  convert_element_type_13 = convert_element_type_14 = None
        convert_element_type_15: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        add_10: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_15);  add_9 = convert_element_type_15 = None
        convert_element_type_16: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.float32);  arg19_1 = None
        add_11: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_10, convert_element_type_16);  add_10 = convert_element_type_16 = None
        convert_element_type_17: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.float32);  arg20_1 = None
        add_12: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_17);  add_11 = convert_element_type_17 = None
        convert_element_type_18: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.float32);  arg21_1 = None
        add_13: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_18);  add_12 = convert_element_type_18 = None
        convert_element_type_19: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        add_14: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_19);  add_13 = convert_element_type_19 = None
        view_5: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg23_1, _shape_param_8);  arg23_1 = _shape_param_8 = None
        convert_element_type_20: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        mul_3: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_4: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, mul_3);  view_5 = mul_3 = None
        convert_element_type_21: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.float32);  mul_4 = None
        full_3: "f32[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_9, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        view_6: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg25_1, _shape_param_10);  arg25_1 = _shape_param_10 = None
        permute_2: "f32[6, 128, 128]" = torch.ops.aten.permute.default(arg26_1, [2, 0, 1]);  arg26_1 = None
        unsqueeze_3: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_2, 0);  permute_2 = None
        add_15: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_3, full_3);  unsqueeze_3 = full_3 = None
        add_16: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_6, add_15);  view_6 = add_15 = None
        convert_element_type_22: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        convert_element_type_23: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        sub_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_23, arg27_1);  convert_element_type_23 = arg27_1 = None
        exp_1: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_1: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, arg28_1);  exp_1 = arg28_1 = None
        mul_5: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_21, div_1);  convert_element_type_21 = None
        sum_3: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [-1], True)
        neg_1: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_1: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_3, mul_5);  neg_1 = sum_3 = mul_5 = None
        convert_element_type_24: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_7: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_24, _shape_param_11);  convert_element_type_24 = _shape_param_11 = None
        view_8: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_7, _shape_param_12);  view_7 = _shape_param_12 = None
        view_9: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_8, _shape_param_13);  _shape_param_13 = None
        convert_element_type_25: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        add_17: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_25);  add_14 = convert_element_type_25 = None
        sum_4: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_17, [0], True, dtype = torch.float32);  add_17 = None
        squeeze_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_4, 0);  sum_4 = None
        permute_3: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        _unsafe_masked_index_put_accumulate_1: "f32[32, 6]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, full_1, [arg29_1], permute_3);  full_2 = full_1 = arg29_1 = permute_3 = None
        return (view_4, _unsafe_masked_index_put_accumulate, view_9, _unsafe_masked_index_put_accumulate_1)



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
