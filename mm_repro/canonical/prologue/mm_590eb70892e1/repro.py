"""Curated MM repro. Pattern hash: 590eb70892e1."""
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
    def forward(self, arg0_1: "f32[8, 1]", arg1_1: "f32[8, 2]", arg2_1: "f32[8, 1]", arg3_1: "f32[8, 2]", arg4_1: "bf16[8, 2]", arg5_1: "i64[8]", arg6_1: "i64[8]", arg7_1: "bf16[8192, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        log: "f32[8, 1]" = torch.ops.aten.log.default(arg0_1);  arg0_1 = None
        sub: "f32[8, 2]" = torch.ops.aten.sub.Tensor(arg1_1, log);  arg1_1 = log = None
        convert_element_type: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        convert_element_type_1: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(convert_element_type_1);  convert_element_type_1 = None
        mul: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp, arg2_1);  exp = arg2_1 = None
        sub_1: "f32[8, 2]" = torch.ops.aten.sub.Tensor(arg3_1, mul);  arg3_1 = mul = None
        convert_element_type_2: "bf16[8, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        add: "bf16[8, 2]" = torch.ops.aten.add.Tensor(arg4_1, convert_element_type_2);  arg4_1 = convert_element_type_2 = None
        full: "bf16[8, 1024, 2]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "bf16[8, 1024, 2]" = torch.ops.aten.index_put.default(full, [arg5_1, arg6_1], add, True);  full = arg5_1 = arg6_1 = add = None
        view: "bf16[8192, 2]" = torch.ops.aten.view.default(index_put, _shape_param_1);  index_put = _shape_param_1 = None
        permute: "bf16[2, 8192]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[2, 768]" = torch.ops.aten.mm.default(permute, arg7_1);  permute = arg7_1 = None
        convert_element_type_3: "f32[2, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_3

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
