"""Curated MM repro. Pattern hash: dcae930cfb96."""
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
    def forward(self, arg0_1: "bf16[256, 512, 64]", arg1_1: "bf16[256, 64, 1024]", arg2_1: "bf16[16, 16, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        view_2: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(permute, _shape_param_1);  permute = _shape_param_1 = None
        slice_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_2, 2, 1, 9223372036854775807);  view_2 = None
        view_3: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_3, [None, None, None, iota]);  view_3 = iota = None
        add: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(arg2_1, index);  arg2_1 = index = None
        add_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add, 0);  add = None
        mul: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_1, 0.125)
        convert_element_type: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        eq: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type, convert_element_type)
        abs_1: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type)
        ne: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        mul_1: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_1);  mul_1 = None
        convert_element_type_1: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        mul_2: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        return (convert_element_type, logical_not, mul_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
