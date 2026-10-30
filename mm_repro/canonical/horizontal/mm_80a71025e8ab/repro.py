"""Curated MM repro. Pattern hash: 80a71025e8ab."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "f32[4096, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "f32[4096, 4096]", arg5_1: "bf16[4096, 4096]", arg6_1: "f32[4096, 4096]", arg7_1: "f32[4096, 4096]", arg8_1: "f32[4096, 4096]", arg9_1: "f32[4096, 4096]", arg10_1: "f32[4096, 4096]", arg11_1: "f32[4096, 4096]"):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = None
        convert_element_type: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        add: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        mm_1: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg3_1, arg1_1);  arg3_1 = None
        convert_element_type_1: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        add_1: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(arg4_1, convert_element_type_1);  arg4_1 = convert_element_type_1 = None
        mm_2: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None
        convert_element_type_2: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        add_2: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_2);  arg6_1 = convert_element_type_2 = None
        add_3: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add, arg7_1);  add = arg7_1 = None
        add_4: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_1, arg8_1);  add_1 = arg8_1 = None
        add_5: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_2, arg9_1);  add_2 = arg9_1 = None
        add_6: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_3, arg10_1);  add_3 = arg10_1 = None
        add_7: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_4, arg11_1);  add_4 = arg11_1 = None
        return (add_5, add_6, add_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
