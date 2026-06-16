"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: 02fe2c82b1fd
Shape hash: c5abdf2a
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "i64[1, 128]", arg2_1: "i64[32, 128]", arg3_1: "f32[]", arg4_1: "bf16[512, 128, 128]", arg5_1: "b8[1, 1, 2048, 2048]", arg6_1: "f32[32, 16, 128, 1]", arg7_1: "f32[32, 16, 128, 1]", arg8_1: "bf16[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        unsqueeze_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg1_1, 1);  arg1_1 = None
        unsqueeze_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg2_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg2_1, [unsqueeze_2, unsqueeze_5]);  arg2_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_1);  bitwise_and_1 = _shape_param_1 = None
        full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, arg3_1, full_1);  expand = arg3_1 = full_1 = None
        view_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg4_1, _shape_param_2);  arg4_1 = _shape_param_2 = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg5_1, 2, 0, 128);  arg5_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view_1, full_2);  view_1 = full_2 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = where = None
        sub: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add, arg6_1);  add = arg6_1 = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg7_1);  exp = arg7_1 = None
        mul: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_1: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        where_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, convert_element_type_1, arg8_1);  slice_2 = convert_element_type_1 = arg8_1 = None
        view_2: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_2, _shape_param_3);  where_2 = _shape_param_3 = None
        return view_2



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
