"""Curated MM repro. Pattern hash: ac2438677e31."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4096, 49, 32]", arg1_1: "bf16[4096, 32, 49]", arg2_1: "i64[49, 49]", arg3_1: "f32[169, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        bmm: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_0);  _shape_param_0 = None
        view_1: "i64[2401]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        index: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg3_1, [view_1]);  arg3_1 = view_1 = None
        view_2: "f32[49, 49, 32]" = torch.ops.aten.view.default(index, _shape_param_1);  index = _shape_param_1 = None
        permute: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_2, [2, 0, 1]);  view_2 = None
        clone: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        add: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view, unsqueeze);  view = unsqueeze = None
        amax: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add, [-1], True)
        sub: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add, amax);  add = None
        exp: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[128, 32, 49, 49]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_2);  _shape_param_2 = None
        view_3: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(expand, _shape_param_3);  expand = _shape_param_3 = None
        constant_pad_nd: "bf16[4096, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        return (bmm, amax, sum_1, convert_element_type, constant_pad_nd)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
