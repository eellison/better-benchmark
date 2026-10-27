"""
Standalone repro captured via capture_hook.
Label: GoogleFnet
Pattern hash: 539f543ae407
Shape hash: 1bf21091
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
    def forward(self, arg0_1: "f32[16384, 32000]", arg1_1: "f32[16384, 768]", arg2_1: "f32[32, 512, 768, 2]", arg3_1: "f32[32, 512, 768]", arg4_1: "b8[32, 512, 768]", arg5_1: "f32[768, 768]", arg6_1: "f32[16384, 768]", arg7_1: "f32[768]", arg8_1: "f32[32, 512, 768]", arg9_1: "f32[32, 512, 1]", arg10_1: "i64[1, 512]", arg11_1: "i64[1, 512]", arg12_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        permute: "f32[32000, 16384]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "f32[32000, 768]" = torch.ops.aten.mm.default(permute, arg1_1);  permute = arg1_1 = None
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg2_1, 3, 0);  arg2_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg3_1, select);  arg3_1 = select = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, mul);  add = mul = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_1: "f32[16384, 768]" = torch.ops.aten.mm.default(view, permute_2);  permute_2 = None
        permute_3: "f32[768, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_2: "f32[768, 768]" = torch.ops.aten.mm.default(permute_3, arg6_1);  permute_3 = arg6_1 = None
        sum_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view, [0], True);  view = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        view_2: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_2, arg7_1);  arg7_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg8_1);  mul_2 = None
        sum_3: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg8_1, sum_3);  sum_3 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_2);  mul_3 = sum_2 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_5);  sub = mul_5 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg9_1, sub_1);  arg9_1 = sub_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_2, arg8_1);  arg8_1 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_2, [0, 1]);  view_2 = None
        sum_6: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_6, [0], True)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg10_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg10_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg10_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg10_1], sum_6);  full = unsqueeze = arg10_1 = sum_6 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg11_1, _shape_param_4);  arg11_1 = _shape_param_4 = None
        ge_1: "b8[32, 512]" = torch.ops.aten.ge.Scalar(expand, 0)
        lt_1: "b8[32, 512]" = torch.ops.aten.lt.Scalar(expand, 4)
        bitwise_and_2: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[32, 512]" = torch.ops.aten.ne.Scalar(expand, -1)
        bitwise_and_3: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[4, 768]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        _unsafe_masked_index_put_accumulate_1: "f32[4, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [expand], mul_6);  full_1 = unsqueeze_1 = expand = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg12_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg12_1, 32000)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg12_1, 3)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_2);  bitwise_and_4 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_2: "f32[32000, 768]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        _unsafe_masked_index_put_accumulate_2: "f32[32000, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_2, [arg12_1], mul_6);  full_2 = unsqueeze_2 = arg12_1 = mul_6 = None
        add_1: "f32[32000, 768]" = torch.ops.aten.add.Tensor(mm, _unsafe_masked_index_put_accumulate_2);  mm = _unsafe_masked_index_put_accumulate_2 = None
        return (mm_2, view_1, sum_4, sum_5, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_1)



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
