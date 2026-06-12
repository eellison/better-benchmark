"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 0e08b9d50286
Shape hash: 05376402
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
    def forward(self, arg0_1: "b8[8, 1024, 768]", arg1_1: "f32[8, 1024, 768]", arg2_1: "f32[768]", arg3_1: "f32[8, 1024, 768]", arg4_1: "f32[8, 1024, 1]", arg5_1: "i64[8, 1024]", arg6_1: "i64[8, 1024]", arg7_1: "b8[8, 1024, 1]", arg8_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        convert_element_type: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  arg2_1 = None
        mul_3: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = None
        sum_2: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_1);  mul_3 = sum_1 = None
        sub_1: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  arg3_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1, [0, 1]);  mul_1 = None
        full: "b8[8, 1024, 1]" = torch.ops.aten.full.default(_shape_param_0, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        full_1: "f32[1, 768]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        _unsafe_masked_index_put_accumulate: "f32[1, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, full, [arg5_1], mul_6);  full_1 = full = arg5_1 = None
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg6_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg6_1, 4098)
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 1)
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[4098, 768]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        _unsafe_masked_index_put_accumulate_1: "f32[4098, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze, [arg6_1], mul_6);  full_2 = unsqueeze = arg6_1 = None
        full_3: "f32[50265, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate_2: "f32[50265, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, arg7_1, [arg8_1], mul_6);  full_3 = arg7_1 = arg8_1 = mul_6 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate_2)



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
