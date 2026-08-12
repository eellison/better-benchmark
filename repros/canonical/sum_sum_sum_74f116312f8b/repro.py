"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 74f116312f8b
Shape hash: 40f378b3
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
    def forward(self, arg0_1: "bf16[128104, 1536]", arg1_1: "bf16[4096, 1536]", arg2_1: "f32[8, 512, 1536]", arg3_1: "bf16[4096, 1536]", arg4_1: "bf16[4096, 1536]", arg5_1: "b8[8, 512, 1536]", arg6_1: "f32[1536]", arg7_1: "f32[8, 512, 1536]", arg8_1: "f32[1, 512, 1536]", arg9_1: "f32[8, 512, 1]", arg10_1: "f32[8, 512, 1]", arg11_1: "i64[1, 512]", arg12_1: "i64[8, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        slice_1: "bf16[128100, 1536]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, -4);  arg0_1 = None
        convert_element_type: "f32[128100, 1536]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        view: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        convert_element_type_1: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type_1);  arg2_1 = convert_element_type_1 = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        convert_element_type_2: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        view_2: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        convert_element_type_3: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_2, mul);  add_2 = mul = None
        mul_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1, arg6_1);  arg6_1 = None
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, 1536)
        sum_1: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        add_3: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(arg7_1, arg8_1);  arg7_1 = arg8_1 = None
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_3, arg9_1);  add_3 = arg9_1 = None
        mul_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, arg10_1);  sub = None
        mul_5: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, mul_4);  mul_2 = None
        sum_2: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_5, [2], True);  mul_5 = None
        mul_6: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_4, sum_2);  sum_2 = None
        sub_1: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_1, mul_6);  sub_1 = mul_6 = None
        div: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg10_1, 1536);  arg10_1 = None
        mul_7: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1, mul_4);  mul_4 = None
        sum_3: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_8, [0, 1]);  mul_8 = None
        sum_4: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        sum_5: "f32[1, 512, 1536]" = torch.ops.aten.sum.dim_IntList(mul_7, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg11_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg11_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg11_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 1536]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 1536]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg11_1], sum_5);  full = unsqueeze = arg11_1 = sum_5 = None
        ge_1: "b8[8, 512]" = torch.ops.aten.ge.Scalar(arg12_1, 0)
        lt_1: "b8[8, 512]" = torch.ops.aten.lt.Scalar(arg12_1, 128100)
        bitwise_and_2: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[8, 512]" = torch.ops.aten.ne.Scalar(arg12_1, 0)
        bitwise_and_3: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[128100, 1536]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate_1: "f32[128100, 1536]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [arg12_1], mul_7);  full_1 = unsqueeze_1 = arg12_1 = mul_7 = None
        add_4: "f32[128100, 1536]" = torch.ops.aten.add.Tensor(convert_element_type, _unsafe_masked_index_put_accumulate_1);  convert_element_type = _unsafe_masked_index_put_accumulate_1 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, add_4)



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
