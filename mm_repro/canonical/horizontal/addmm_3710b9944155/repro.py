"""Curated MM repro. Pattern hash: 3710b9944155."""
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
    def forward(self, arg0_1: "bf16[16, 128, 768]", arg1_1: "bf16[16, 128, 1]", arg2_1: "bf16[768]", arg3_1: "f32[16, 128, 1]", arg4_1: "bf16[768]", arg5_1: "bf16[768, 768]", arg6_1: "bf16[768]", arg7_1: "bf16[768, 768]", arg8_1: "bf16[768]", arg9_1: "bf16[768, 768]", arg10_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        sub: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sub);  arg2_1 = sub = None
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(arg3_1);  arg3_1 = None
        convert_element_type: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt, torch.bfloat16);  sqrt = None
        add: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type, 1e-06);  convert_element_type = None
        div: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add);  mul = add = None
        add_1: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg4_1);  div = arg4_1 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_0);  _shape_param_0 = None
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg6_1, view, arg5_1);  arg6_1 = view = arg5_1 = None
        view_2: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        permute_1: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_4: "bf16[2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_3);  _shape_param_3 = None
        addmm_1: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view_4, arg7_1);  arg8_1 = view_4 = arg7_1 = None
        view_6: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        permute_3: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        clone_1: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_7: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        view_8: "bf16[2048, 768]" = torch.ops.aten.view.default(add_1, _shape_param_6);  add_1 = _shape_param_6 = None
        addmm_2: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_8, arg9_1);  arg10_1 = view_8 = arg9_1 = None
        view_10: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(addmm_2, _shape_param_7);  addmm_2 = _shape_param_7 = None
        permute_6: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_11: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_8);  clone_2 = _shape_param_8 = None
        return (view_3, view_7, view_11)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
