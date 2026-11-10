"""Curated MM repro. Pattern hash: 568da462388f."""
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 16384]", arg2_1: "bf16[4096, 16384]", arg3_1: "bf16[4096, 16384]", arg4_1: "bf16[4096, 16384]", arg5_1: "bf16[4096, 16384]", arg6_1: "bf16[4096, 16384]", arg7_1: "bf16[4096, 16384]", arg8_1: "bf16[4096, 16384]", arg9_1: "bf16[4096, 16384]", arg10_1: "bf16[4096, 16384]", arg11_1: "bf16[4096, 16384]", arg12_1: "bf16[4096, 16384]"):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_1: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        add: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        convert_element_type_2: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        convert_element_type_3: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        add_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        convert_element_type_4: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        add_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_2, convert_element_type_4);  add_2 = convert_element_type_4 = None
        convert_element_type_5: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float32);  arg6_1 = None
        add_4: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_3, convert_element_type_5);  add_3 = convert_element_type_5 = None
        convert_element_type_6: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        add_5: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_6);  add_4 = convert_element_type_6 = None
        convert_element_type_7: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_6: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_7);  add_5 = convert_element_type_7 = None
        convert_element_type_8: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        add_7: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_8);  add_6 = convert_element_type_8 = None
        convert_element_type_9: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.float32);  arg10_1 = None
        add_8: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_9);  add_7 = convert_element_type_9 = None
        convert_element_type_10: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.float32);  arg11_1 = None
        add_9: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_10);  add_8 = convert_element_type_10 = None
        convert_element_type_11: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        add_10: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_11);  add_9 = convert_element_type_11 = None
        return add_10

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
