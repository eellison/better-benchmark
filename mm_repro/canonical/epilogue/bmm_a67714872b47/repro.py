"""Curated MM repro. Pattern hash: a67714872b47."""
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
    def forward(self, arg0_1: "bf16[8192, 49, 32]", arg1_1: "bf16[8192, 32, 49]", arg2_1: "bf16[8192, 56, 56]", arg3_1: "i64[49, 49]", arg4_1: "f32[169, 16]", arg5_1: "f32[512, 16, 49, 1]", arg6_1: "f32[512, 16, 49, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        bmm: "bf16[8192, 49, 49]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        slice_1: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, -7);  arg2_1 = None
        slice_2: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_1: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_1);  slice_2 = _shape_param_1 = None
        view_2: "i64[2401]" = torch.ops.aten.view.default(arg3_1, [-1]);  arg3_1 = None
        index: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg4_1, [view_2]);  arg4_1 = view_2 = None
        view_3: "f32[49, 49, 16]" = torch.ops.aten.view.default(index, _shape_param_2);  index = _shape_param_2 = None
        permute: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_3, [2, 0, 1]);  view_3 = None
        clone: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "f32[1, 16, 49, 49]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        add: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_1, unsqueeze);  view_1 = unsqueeze = None
        sub: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add, arg5_1);  add = arg5_1 = None
        exp: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = div = None
        return mul

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
