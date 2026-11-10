"""Curated MM repro. Pattern hash: 3e75e0932cf2."""
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
    def forward(self, arg0_1: "f32[32, 128, 1]", arg1_1: "f32[32, 128, 2048]", arg2_1: "f32[32, 128, 1]", arg3_1: "f32[2048]", arg4_1: "f32[2048]", arg5_1: "f32[2048, 2048]", arg6_1: "f32[2048, 2048]", arg7_1: "f32[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-05);  arg0_1 = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(arg1_1, arg2_1);  arg1_1 = arg2_1 = None
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_1: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, convert_element_type);  convert_element_type = None
        convert_element_type_2: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, convert_element_type_2);  convert_element_type_2 = None
        convert_element_type_3: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        mm_2: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, convert_element_type_3);  view = convert_element_type_3 = None
        view_4: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        permute_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(mm_2, _shape_param_3);  mm_2 = _shape_param_3 = None
        permute_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        convert_element_type_4: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        permute_6: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_4, [0, 1, 3, 2]);  convert_element_type_4 = None
        convert_element_type_5: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_6, torch.bfloat16);  permute_6 = None
        convert_element_type_6: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_3, torch.bfloat16);  permute_3 = None
        clone: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(convert_element_type_6, memory_format = torch.contiguous_format);  convert_element_type_6 = None
        view_7: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        clone_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_8: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        permute_7: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1])
        return (view_7, convert_element_type_5, view_8, permute_7)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
