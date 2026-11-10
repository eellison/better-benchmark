"""Curated MM repro. Pattern hash: 154822c54090."""
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
    def forward(self, arg0_1: "f32[2048, 8, 49, 49]", arg1_1: "i64[2401]", arg2_1: "bf16[16384, 49, 32]", arg3_1: "bf16[16384, 32, 49]", arg4_1: "bf16[16384, 56, 56]", arg5_1: "i64[49, 49]", arg6_1: "f32[169, 8]", arg7_1: "f32[2048, 8, 49, 1]", arg8_1: "f32[2048, 8, 49, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        sum_1: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0], True, dtype = torch.float32);  arg0_1 = None
        squeeze: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_1, 0);  sum_1 = None
        permute: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        view: "f32[2401, 8]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        full: "f32[169, 8]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        index_put: "f32[169, 8]" = torch.ops.aten.index_put.default(full, [arg1_1], view, True);  arg1_1 = view = None
        bmm: "bf16[16384, 49, 49]" = torch.ops.aten.bmm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view_1: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        convert_element_type: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        slice_1: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 0, -7);  arg4_1 = None
        slice_2: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_2: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_3);  slice_2 = _shape_param_3 = None
        view_3: "i64[2401]" = torch.ops.aten.view.default(arg5_1, [-1]);  arg5_1 = None
        index: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg6_1, [view_3]);  arg6_1 = None
        view_4: "f32[49, 49, 8]" = torch.ops.aten.view.default(index, _shape_param_4);  index = _shape_param_4 = None
        permute_1: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_4, [2, 0, 1]);  view_4 = None
        clone: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        unsqueeze: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        add: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_2, unsqueeze);  view_2 = unsqueeze = None
        sub: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add, arg7_1);  add = arg7_1 = None
        exp: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp, arg8_1);  exp = arg8_1 = None
        mul: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = None
        sum_2: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg, sum_2, mul);  neg = sum_2 = mul = None
        convert_element_type_1: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        sum_3: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True, dtype = torch.float32);  fma = None
        squeeze_1: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_3, 0);  sum_3 = None
        permute_2: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        view_5: "f32[2401, 8]" = torch.ops.aten.view.default(permute_2, _shape_param_5);  permute_2 = _shape_param_5 = None
        index_put_1: "f32[169, 8]" = torch.ops.aten.index_put.default(full, [view_3], view_5, True);  full = view_3 = view_5 = None
        view_6: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_6);  convert_element_type_1 = _shape_param_6 = None
        return (index_put, index_put_1, view_6)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
