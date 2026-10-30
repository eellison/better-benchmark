"""Curated MM repro. Pattern hash: 6b98afc0fb21."""
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
    def forward(self, arg0_1: "f32[32, 128, 1]", arg1_1: "bf16[32, 128, 512]", arg2_1: "bf16[512]", arg3_1: "bf16[512, 384]", arg4_1: "bf16[512, 384]", arg5_1: "bf16[512, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(arg0_1, 1e-06);  arg0_1 = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg1_1, rsqrt);  arg1_1 = rsqrt = None
        convert_element_type: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, convert_element_type);  arg2_1 = convert_element_type = None
        view: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  _shape_param_0 = None
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(view, arg3_1);  view = arg3_1 = None
        view_2: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        permute_1: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        view_4: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_3);  _shape_param_3 = None
        mm_1: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_4, arg4_1);  view_4 = arg4_1 = None
        view_6: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        permute_3: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        clone_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_7: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_5);  clone_1 = _shape_param_5 = None
        view_8: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_6);  mul_1 = _shape_param_6 = None
        mm_2: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_8, arg5_1);  view_8 = arg5_1 = None
        view_10: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        permute_6: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_11: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_8);  clone_2 = _shape_param_8 = None
        return (view_3, view_7, view_11)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
