"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: d915970dff62
Shape hash: 00333af8
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
    def forward(self, arg0_1: "f32[54]", arg1_1: "bf16[16384, 54]", arg2_1: "f32[98304, 1, 1]", arg3_1: "f32[98304, 1, 1]", arg4_1: "bf16[98304, 64, 1]", arg5_1: "bf16[98304, 9, 1]", arg6_1: "i64[9, 512, 1, 1]", arg7_1: "i64[1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        view: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view, convert_element_type);  view = convert_element_type = None
        view_1: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, arg3_1);  exp = arg3_1 = None
        convert_element_type_2: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16)
        expand: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_2);  convert_element_type_2 = _shape_param_2 = None
        permute: "bf16[98304, 1, 9]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
        convert_element_type_3: "f32[98304, 64, 1]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        convert_element_type_4: "f32[98304, 1, 9]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        mul: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_4);  convert_element_type_3 = convert_element_type_4 = None
        convert_element_type_5: "bf16[98304, 64, 9]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_6: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        view_2: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        view_3: "bf16[32, 512, 3456]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        permute_1: "bf16[32, 3456, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        convert_element_type_7: "f32[32, 3456, 512]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        view_4: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_5);  convert_element_type_7 = _shape_param_5 = None
        permute_2: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_4, [0, 1, 2, 4, 3, 5]);  view_4 = None
        full: "f32[32, 384, 520, 1]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        index_put: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full, [None, None, arg6_1, arg7_1], permute_2, True);  arg6_1 = arg7_1 = permute_2 = None
        constant_pad_nd: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put, _shape_param_7, 0.0);  index_put = _shape_param_7 = None
        convert_element_type_8: "bf16[32, 384, 512, 1]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        squeeze: "bf16[32, 384, 512]" = torch.ops.aten.squeeze.dim(convert_element_type_8, -1);  convert_element_type_8 = None
        permute_3: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_5: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_8);  clone = _shape_param_8 = None
        permute_4: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_1: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[384]" = torch.ops.aten.view.default(sum_1, _shape_param_9);  sum_1 = _shape_param_9 = None
        convert_element_type_9: "bf16[384]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_10: "f32[384]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        mul_1: "f32[98304, 9, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_6, div);  convert_element_type_6 = None
        sum_2: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1, [1], True)
        neg: "f32[98304, 9, 1]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[98304, 9, 1]" = torch.ops.prims.fma.default(neg, sum_2, mul_1);  neg = sum_2 = mul_1 = None
        convert_element_type_11: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_7: "bf16[32, 512, 54]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_10);  convert_element_type_11 = _shape_param_10 = None
        sum_3: "f32[1, 1, 54]" = torch.ops.aten.sum.dim_IntList(view_7, [0, 1], True, dtype = torch.float32)
        view_8: "f32[54]" = torch.ops.aten.view.default(sum_3, _shape_param_11);  sum_3 = _shape_param_11 = None
        convert_element_type_12: "bf16[54]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[16384, 54]" = torch.ops.aten.view.default(view_7, _shape_param_12);  view_7 = _shape_param_12 = None
        permute_5: "bf16[54, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0])
        convert_element_type_13: "f32[54]" = torch.ops.prims.convert_element_type.default(convert_element_type_12, torch.float32);  convert_element_type_12 = None
        return (full, view_5, permute_4, convert_element_type_10, view_9, permute_5, convert_element_type_13)



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
