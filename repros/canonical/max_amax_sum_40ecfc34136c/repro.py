"""
Standalone repro captured via capture_hook.
Label: hf_openai/gpt-oss-20b_infer
Pattern hash: 40ecfc34136c
Shape hash: d9b19a63
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
    def forward(self, arg0_1: "bf16[64, 1000, 1000]", arg1_1: "bf16[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[1000]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[1000]" = torch.ops.prims.iota.default(1000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[1000]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 1000]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 1000]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1000, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        sub: "i64[1, 1, 1000, 1]" = torch.ops.aten.sub.Tensor(unsqueeze_5, 128)
        gt: "b8[1, 1, 1000, 1000]" = torch.ops.aten.gt.Tensor(unsqueeze_2, sub);  sub = None
        bitwise_and: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(full, gt);  full = gt = None
        le: "b8[1, 1, 1000, 1000]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5);  unsqueeze_2 = unsqueeze_5 = None
        bitwise_and_1: "b8[1, 1, 1000, 1000]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, le);  bitwise_and = le = None
        expand: "b8[1, 1, 1000, 1000]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_1);  bitwise_and_1 = _shape_param_1 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 1000, 1000]" = torch.ops.aten.where.self(expand, full_1, full_2);  expand = full_1 = full_2 = None
        add_2: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul, where);  mul = None
        view_1: "bf16[1, 64, 1, 1]" = torch.ops.aten.view.default(arg1_1, [1, -1, 1, 1]);  arg1_1 = None
        expand_1: "bf16[1, 64, 1000, 1]" = torch.ops.aten.expand.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        cat: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add_2, expand_1], -1);  add_2 = expand_1 = None
        max_1 = torch.ops.aten.max.dim(cat, -1, True)
        getitem: "bf16[1, 64, 1000, 1]" = max_1[0]
        getitem_1: "i64[1, 64, 1000, 1]" = max_1[1];  max_1 = getitem_1 = None
        sub_1: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(cat, getitem);  cat = getitem = None
        convert_element_type: "f32[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(sub_1, torch.float32);  sub_1 = None
        amax: "f32[1, 64, 1000, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub_2: "f32[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[1, 64, 1000, 1001]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[1, 64, 1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 64, 1000, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        slice_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 3, 0, -1);  convert_element_type_1 = None
        clone: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        expand_2: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.expand.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        view_2: "bf16[64, 1000, 1000]" = torch.ops.aten.view.default(expand_2, _shape_param_4);  expand_2 = _shape_param_4 = None
        return (where, view_2)



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
