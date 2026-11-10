"""Curated MM repro. Pattern hash: 450bb0e98288."""
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
    def forward(self, arg0_1: "bf16[401408, 512]", arg1_1: "bf16[512, 128]", arg2_1: "f32[128]", arg3_1: "f32[128, 3136, 128]", arg4_1: "f32[128, 3136, 1]", arg5_1: "f32[128, 3136, 128]", arg6_1: "b8[128, 1, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        mm: "bf16[401408, 128]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg2_1);  arg2_1 = None
        mul_1: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul, 128)
        sum_1: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, arg3_1);  arg3_1 = None
        sum_3: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1]);  convert_element_type = None
        add: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(arg5_1, mul_4);  arg5_1 = mul_4 = None
        view_1: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type_1: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16)
        convert_element_type_2: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        div: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9956521736457944);  convert_element_type_2 = None
        mul_6: "bf16[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, div);  convert_element_type_1 = div = None
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_1, 56);  add_1 = None
        index: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(mul_6, [None, None, fmod]);  mul_6 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index, [None, fmod]);  index = fmod = None
        view_2: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_1, _shape_param_2);  index_1 = _shape_param_2 = None
        permute: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2, 4, 5]);  view_2 = None
        clone: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_3: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        view_4: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        view_5: "bf16[401408, 128]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_1: "bf16[128, 401408]" = torch.ops.aten.permute.default(view_5, [1, 0])
        sum_5: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32)
        view_6: "f32[128]" = torch.ops.aten.view.default(sum_5, _shape_param_6);  sum_5 = _shape_param_6 = None
        convert_element_type_3: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_6, torch.bfloat16);  view_6 = None
        convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (sum_3, sum_4, view_1, view_5, permute_1, convert_element_type_4)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
