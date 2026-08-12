"""
Standalone repro captured via capture_hook.
Label: hf_openai/gpt-oss-20b_infer
Pattern hash: 4053b14a3062
Shape hash: 038e722e
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
    def forward(self, arg0_1: "bf16[32, 2880]", arg1_1: "i64[4000]", arg2_1: "bf16[4000, 2880]", arg3_1: "bf16[1000, 4]", arg4_1: "i64[4000]", arg5_1: "b8[4000, 1]", arg6_1: "bf16[1, 1000, 2880]", arg7_1: "bf16[2880]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg0_1, [arg1_1]);  arg0_1 = arg1_1 = None
        add: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(arg2_1, index);  arg2_1 = index = None
        convert_element_type: "f32[1000, 4]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        amax: "f32[1000, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[1000, 4]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[1000, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[1000, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[1000, 4]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view: "bf16[4000]" = torch.ops.aten.view.default(convert_element_type_1, [-1]);  convert_element_type_1 = None
        index_1: "bf16[4000]" = torch.ops.aten.index.Tensor(view, [arg4_1]);  view = None
        unsqueeze: "bf16[4000, 1]" = torch.ops.aten.unsqueeze.default(index_1, -1);  index_1 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add, unsqueeze);  add = unsqueeze = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(arg5_1, full, mul);  arg5_1 = full = mul = None
        empty: "i64[4000]" = torch.ops.aten.empty.memory_format(_shape_param_0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        iota: "i64[4000]" = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[4000]" = torch.ops.aten.index_put.default(empty, [arg4_1], iota);  empty = arg4_1 = iota = None
        index_2: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(where, [index_put]);  where = index_put = None
        view_1: "bf16[1000, 4, 2880]" = torch.ops.aten.view.default(index_2, _shape_param_1);  index_2 = _shape_param_1 = None
        sum_2: "bf16[1000, 2880]" = torch.ops.aten.sum.dim_IntList(view_1, [1]);  view_1 = None
        view_2: "bf16[1, 1000, 2880]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        add_1: "bf16[1, 1000, 2880]" = torch.ops.aten.add.Tensor(arg6_1, view_2);  arg6_1 = view_2 = None
        convert_element_type_2: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        pow_1: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_2: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = rsqrt = None
        mul_2: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg7_1, mul_1);  arg7_1 = mul_1 = None
        convert_element_type_3: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        view_3: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        return view_3



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
