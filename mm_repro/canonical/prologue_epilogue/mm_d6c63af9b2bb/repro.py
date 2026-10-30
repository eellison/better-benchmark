"""Curated MM repro. Pattern hash: d6c63af9b2bb."""
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
    def forward(self, arg0_1: "f32[32, 1]", arg1_1: "f32[32, 2]", arg2_1: "f32[32, 1]", arg3_1: "f32[32, 2]", arg4_1: "bf16[32, 2]", arg5_1: "i64[32]", arg6_1: "i64[32]", arg7_1: "bf16[2, 2048]", arg8_1: "f32[2048]", arg9_1: "f32[32, 128, 2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        log: "f32[32, 1]" = torch.ops.aten.log.default(arg0_1);  arg0_1 = None
        sub: "f32[32, 2]" = torch.ops.aten.sub.Tensor(arg1_1, log);  arg1_1 = log = None
        convert_element_type: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        convert_element_type_1: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        exp: "f32[32, 2]" = torch.ops.aten.exp.default(convert_element_type_1);  convert_element_type_1 = None
        mul: "f32[32, 2]" = torch.ops.aten.mul.Tensor(exp, arg2_1);  exp = arg2_1 = None
        sub_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(arg3_1, mul);  arg3_1 = mul = None
        convert_element_type_2: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        add: "bf16[32, 2]" = torch.ops.aten.add.Tensor(arg4_1, convert_element_type_2);  arg4_1 = convert_element_type_2 = None
        full: "bf16[32, 128, 2]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "bf16[32, 128, 2]" = torch.ops.aten.index_put.default(full, [arg5_1, arg6_1], add, True);  full = arg5_1 = arg6_1 = add = None
        view: "bf16[4096, 2]" = torch.ops.aten.view.default(index_put, _shape_param_1);  index_put = _shape_param_1 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, arg7_1);  view = arg7_1 = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_3: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg8_1);  arg8_1 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_1, 2048)
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg9_1);  arg9_1 = None
        return (convert_element_type_3, mul_1, mul_2, mul_3)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
