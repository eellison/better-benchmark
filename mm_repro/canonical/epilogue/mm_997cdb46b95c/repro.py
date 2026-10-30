"""Curated MM repro. Pattern hash: 997cdb46b95c."""
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
    def forward(self, arg0_1: "bf16[16384, 768]", arg1_1: "bf16[768, 768]", arg2_1: "f32[54]", arg3_1: "bf16[16384, 54]", arg4_1: "f32[98304, 1, 1]", arg5_1: "f32[98304, 1, 1]", arg6_1: "f32[32, 384, 520, 1]", arg7_1: "i64[9, 512, 1, 1]", arg8_1: "i64[1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # No stacktrace found for following nodes
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        slice_1: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(view_1, 2, 6, 12);  view_1 = None
        view_2: "bf16[16384, 384]" = torch.ops.aten.view.default(slice_2, _shape_param_1);  slice_2 = _shape_param_1 = None
        permute: "bf16[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None
        clone: "bf16[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        clone_1: "bf16[16384, 384]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        view_3: "bf16[98304, 64, 1]" = torch.ops.aten.view.default(clone_1, _shape_param_2);  clone_1 = _shape_param_2 = None
        convert_element_type: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_4: "bf16[32, 512, 54]" = torch.ops.aten.view.default(arg3_1, _shape_param_3);  arg3_1 = _shape_param_3 = None
        add: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_4, convert_element_type);  view_4 = convert_element_type = None
        view_5: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        convert_element_type_1: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, arg4_1);  convert_element_type_1 = arg4_1 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        convert_element_type_2: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        permute_1: "bf16[98304, 1, 9]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 2, 1]);  convert_element_type_2 = None
        convert_element_type_3: "f32[98304, 64, 1]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32)
        convert_element_type_4: "f32[98304, 1, 9]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        mul: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_4);  convert_element_type_3 = convert_element_type_4 = None
        convert_element_type_5: "bf16[98304, 64, 9]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        view_7: "bf16[32, 512, 3456]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_5);  convert_element_type_5 = _shape_param_5 = None
        permute_2: "bf16[32, 3456, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        convert_element_type_6: "f32[32, 3456, 512]" = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        view_8: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_6);  convert_element_type_6 = _shape_param_6 = None
        permute_3: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(view_8, [0, 1, 2, 4, 3, 5]);  view_8 = None
        index_put: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(arg6_1, [None, None, arg7_1, arg8_1], permute_3, True);  arg6_1 = arg7_1 = arg8_1 = permute_3 = None
        constant_pad_nd: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put, _shape_param_7, 0.0);  index_put = _shape_param_7 = None
        convert_element_type_7: "bf16[32, 384, 512, 1]" = torch.ops.prims.convert_element_type.default(constant_pad_nd, torch.bfloat16);  constant_pad_nd = None
        squeeze: "bf16[32, 384, 512]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_8);  convert_element_type_7 = _shape_param_8 = None
        permute_4: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(squeeze, [0, 2, 1]);  squeeze = None
        clone_2: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_9: "bf16[16384, 384]" = torch.ops.aten.view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        permute_5: "bf16[384, 16384]" = torch.ops.aten.permute.default(view_9, [1, 0])
        return (clone, view_3, view_9, permute_5)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
