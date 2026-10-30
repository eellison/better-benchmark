"""Curated MM repro. Pattern hash: 067ed1cc9848."""
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
    def forward(self, arg0_1: "bf16[384, 256, 64]", arg1_1: "bf16[384, 64, 768]", arg2_1: "bf16[96, 4, 256, 769]", arg3_1: "bf16[96, 4, 197120]", arg4_1: "b8[8, 1024, 12, 513]", arg5_1: "b8[8, 1024, 1, 1]", arg6_1: "f32[]", arg7_1: "bf16[8, 1024, 12, 513]", arg8_1: "bf16[8, 1024, 1, 513]", arg9_1: "f32[8, 1024, 12, 1]", arg10_1: "f32[8, 1024, 12, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        bmm: "bf16[384, 256, 768]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        squeeze: "bf16[96, 4, 256, 768]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        slice_scatter: "bf16[96, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(arg2_1, squeeze, 3, 0, -1);  arg2_1 = squeeze = None
        view_1: "bf16[96, 4, 196864]" = torch.ops.aten.view.default(slice_scatter, _shape_param_1);  slice_scatter = _shape_param_1 = None
        slice_scatter_1: "bf16[96, 4, 197120]" = torch.ops.aten.slice_scatter.default(arg3_1, view_1, 2, 0, -256);  arg3_1 = view_1 = None
        view_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_1, _shape_param_2);  slice_scatter_1 = _shape_param_2 = None
        constant_pad_nd: "bf16[96, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        view_3: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_4);  constant_pad_nd = _shape_param_4 = None
        permute: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        mul: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute, mul);  permute = mul = None
        clone: "bf16[8, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_1, memory_format = torch.contiguous_format);  mul_1 = None
        convert_element_type_1: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        where: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(arg5_1, arg6_1, convert_element_type_1);  arg5_1 = arg6_1 = convert_element_type_1 = None
        add: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(arg7_1, arg8_1);  arg7_1 = arg8_1 = None
        permute_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add, [0, 2, 1, 3]);  add = None
        permute_6: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        convert_element_type_2: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_6, torch.float32);  permute_6 = None
        clone_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        sub: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_1, arg9_1);  clone_1 = arg9_1 = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, arg10_1);  exp = arg10_1 = None
        mul_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where, div);  where = div = None
        return mul_2

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
