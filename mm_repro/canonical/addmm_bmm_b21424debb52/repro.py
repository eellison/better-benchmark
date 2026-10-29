"""Curated MM repro. Pattern hash: b21424debb52."""
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
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "bf16[768, 768]", arg2_1: "bf16[768]", arg3_1: "bf16[2048, 768]", arg4_1: "bf16[768, 768]", arg5_1: "bf16[768]", arg6_1: "bf16[2048, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        gt: "b8[16, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze: "b8[16, 1, 128]" = torch.ops.aten.view.default(gt, _shape_param_0);  gt = _shape_param_0 = None
        repeat: "b8[16, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_1);  unsqueeze = _shape_param_1 = None
        unsqueeze_1: "b8[16, 1, 128, 128]" = torch.ops.aten.view.default(repeat, _shape_param_2);  repeat = _shape_param_2 = None
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0);  unsqueeze_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg2_1, arg3_1, arg1_1);  arg2_1 = arg3_1 = arg1_1 = None
        view_1: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        permute_1: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_2: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        addmm_1: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg5_1, arg6_1, arg4_1);  arg5_1 = arg6_1 = arg4_1 = None
        view_4: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  addmm_1 = _shape_param_5 = None
        permute_3: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_4: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        clone_1: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_5: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_6);  clone_1 = _shape_param_6 = None
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_2, view_5);  view_2 = view_5 = None
        view_6: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_7);  bmm = _shape_param_7 = None
        div: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_6, 8.0);  view_6 = None
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div);  eq = full = div = None
        convert_element_type: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [3], True)
        sub: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        view_7: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_8);  convert_element_type_1 = _shape_param_8 = None
        return view_7

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
