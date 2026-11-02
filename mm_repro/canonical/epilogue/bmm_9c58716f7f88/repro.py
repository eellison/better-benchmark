"""Curated MM repro. Pattern hash: 9c58716f7f88."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
from mm_replay import make_inputs_from_config, benchmark_repro_safe as benchmark_repro
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[64, 1000, 64]", arg1_1: "bf16[64, 64, 1000]", arg2_1: "bf16[1, 1, 1000, 1000]", arg3_1: "bf16[64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[64, 1000, 1000]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        mul: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view, 0.125);  view = None
        add: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        view_1: "bf16[1, 64, 1, 1]" = torch.ops.aten.view.default(arg3_1, [1, -1, 1, 1]);  arg3_1 = None
        expand: "bf16[1, 64, 1000, 1]" = torch.ops.aten.expand.default(view_1, _shape_param_1);  view_1 = _shape_param_1 = None
        cat: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add, expand], -1);  add = expand = None
        max_1 = torch.ops.aten.max.dim(cat, -1, True)
        getitem: "bf16[1, 64, 1000, 1]" = max_1[0]
        getitem_1: "i64[1, 64, 1000, 1]" = max_1[1];  max_1 = getitem_1 = None
        sub: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(cat, getitem);  cat = getitem = None
        convert_element_type: "f32[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        amax: "f32[1, 64, 1000, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub_1: "f32[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[1, 64, 1000, 1001]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 64, 1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1, 64, 1000, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        slice_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.slice.Tensor(convert_element_type_1, 3, 0, -1);  convert_element_type_1 = None
        clone: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        expand_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.expand.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_2: "bf16[64, 1000, 1000]" = torch.ops.aten.view.default(expand_1, _shape_param_3);  expand_1 = _shape_param_3 = None
        return view_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
