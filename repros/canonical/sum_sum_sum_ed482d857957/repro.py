"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: ed482d857957
Shape hash: f69e9b69
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
    def forward(self, arg0_1: "f32[16384, 768]", arg1_1: "f32[768]", arg2_1: "f32[32, 512, 768]", arg3_1: "f32[32, 512, 1]", arg4_1: "i64[1, 512]", arg5_1: "i64[1, 512]", arg6_1: "i64[32, 512]", arg7_1: "f32[32000, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg1_1);  arg1_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(view, [0, 1]);  view = None
        sum_5: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_4, [0], True)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg4_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg4_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg4_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full, unsqueeze, [arg4_1], sum_5);  full = unsqueeze = arg4_1 = sum_5 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg5_1, _shape_param_2);  arg5_1 = _shape_param_2 = None
        ge_1: "b8[32, 512]" = torch.ops.aten.ge.Scalar(expand, 0)
        lt_1: "b8[32, 512]" = torch.ops.aten.lt.Scalar(expand, 4)
        bitwise_and_2: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[32, 512]" = torch.ops.aten.ne.Scalar(expand, -1)
        bitwise_and_3: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_1: "f32[4, 768]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        _unsafe_masked_index_put_accumulate_1: "f32[4, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_1, unsqueeze_1, [expand], mul_4);  full_1 = unsqueeze_1 = expand = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg6_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg6_1, 32000)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg6_1, 3)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_2);  bitwise_and_4 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_2: "f32[32000, 768]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        _unsafe_masked_index_put_accumulate_2: "f32[32000, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_2, [arg6_1], mul_4);  full_2 = unsqueeze_2 = arg6_1 = mul_4 = None
        add: "f32[32000, 768]" = torch.ops.aten.add.Tensor(arg7_1, _unsafe_masked_index_put_accumulate_2);  arg7_1 = _unsafe_masked_index_put_accumulate_2 = None
        return (sum_3, sum_4, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add)



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
