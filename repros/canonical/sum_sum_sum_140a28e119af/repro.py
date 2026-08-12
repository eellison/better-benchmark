"""
Standalone repro captured via capture_hook.
Label: hf_MegatronBertForCausalLM_train
Pattern hash: 140a28e119af
Shape hash: 8ab09cfc
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
    def forward(self, arg0_1: "bf16[29056, 1024]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[8192, 1024]", arg3_1: "bf16[8192, 1024]", arg4_1: "f32[1024]", arg5_1: "f32[16, 512, 1024]", arg6_1: "f32[16, 512, 1]", arg7_1: "f32[16, 512, 1024]", arg8_1: "b8[16, 512, 1024]", arg9_1: "i64[1, 512]", arg10_1: "i64[16, 512]", arg11_1: "i64[16, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        convert_element_type: "f32[29056, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type_1: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        view_1: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_2: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_2);  convert_element_type_1 = convert_element_type_2 = None
        view_2: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_3: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_1: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_3);  add = convert_element_type_3 = None
        mul: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg4_1);  arg4_1 = None
        mul_1: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, 1024)
        sum_1: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = None
        sum_2: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg5_1, sum_2);  sum_2 = None
        sub: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(arg6_1, sub_1);  arg6_1 = sub_1 = None
        mul_5: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_1, arg5_1);  arg5_1 = None
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_1, [0, 1]);  add_1 = None
        add_2: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(arg7_1, mul_4);  arg7_1 = mul_4 = None
        convert_element_type_4: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_6: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_7: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_2, mul_6);  add_2 = mul_6 = None
        sum_5: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg9_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg9_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg9_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 1024]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg9_1], sum_5);  full = unsqueeze = arg9_1 = sum_5 = None
        full_1: "b8[16, 512, 1]" = torch.ops.aten.full.default(_shape_param_4, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        full_2: "f32[2, 1024]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, full_1, [arg10_1], mul_7);  full_2 = full_1 = arg10_1 = None
        ge_1: "b8[16, 512]" = torch.ops.aten.ge.Scalar(arg11_1, 0)
        lt_1: "b8[16, 512]" = torch.ops.aten.lt.Scalar(arg11_1, 29056)
        bitwise_and_2: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 512]" = torch.ops.aten.ne.Scalar(arg11_1, 0)
        bitwise_and_3: "b8[16, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_3: "f32[29056, 1024]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        _unsafe_masked_index_put_accumulate_2: "f32[29056, 1024]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_1, [arg11_1], mul_7);  full_3 = unsqueeze_1 = arg11_1 = mul_7 = None
        add_3: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_2);  convert_element_type = _unsafe_masked_index_put_accumulate_2 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_3)



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
