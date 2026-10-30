"""Curated MM repro. Pattern hash: 478c63b3f825."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "b8[256, 4, 128, 128]", arg1_1: "f32[256, 4, 128, 128]", arg2_1: "bf16[1024, 128, 32]", arg3_1: "bf16[128, 128]", arg4_1: "bf16[128]", arg5_1: "bf16[256, 128, 128]", arg6_1: "bf16[128]", arg7_1: "bf16[128]", arg8_1: "bf16[128, 512]", arg9_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(arg0_1, -1, True);  arg0_1 = None
        logical_not: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(arg1_1, [3], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(arg1_1, amax);  arg1_1 = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not, full, convert_element_type);  logical_not = full = convert_element_type = None
        view: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(where, _shape_param_1);  where = _shape_param_1 = None
        bmm: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view, arg2_1);  view = arg2_1 = None
        view_1: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        permute: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[32768, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        addmm: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg4_1, view_3, arg3_1);  arg4_1 = view_3 = arg3_1 = None
        view_4: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        add: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_4, arg5_1);  view_4 = arg5_1 = None
        mul: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add, arg6_1);  add = arg6_1 = None
        add_1: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg7_1);  mul = arg7_1 = None
        view_5: "bf16[32768, 128]" = torch.ops.aten.view.default(add_1, _shape_param_5);  _shape_param_5 = None
        addmm_1: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg9_1, view_5, arg8_1);  arg9_1 = view_5 = arg8_1 = None
        view_6: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        relu: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_6);  view_6 = None
        view_7: "bf16[32768, 512]" = torch.ops.aten.view.default(relu, _shape_param_7);  relu = _shape_param_7 = None
        return (add_1, view_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
