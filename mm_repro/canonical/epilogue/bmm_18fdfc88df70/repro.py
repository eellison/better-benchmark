"""Curated MM repro. Pattern hash: 18fdfc88df70."""
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "bf16[512, 128, 128]", arg2_1: "i64[1, 128]", arg3_1: "i64[32, 128]", arg4_1: "f32[]", arg5_1: "bf16[512, 128, 128]", arg6_1: "b8[1, 1, 2048, 2048]", arg7_1: "f32[32, 16, 128, 1]", arg8_1: "f32[32, 16, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[32, 1, 1, 1]" = torch.ops.aten.view.default(iota, _shape_param_1);  iota = _shape_param_1 = None
        unsqueeze_4: "i64[1, 1, 128, 1]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  _shape_param_2 = None
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.view.default(arg2_1, _shape_param_3);  arg2_1 = _shape_param_3 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_5]);  arg3_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        full_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(bitwise_and_1, arg4_1, full_1);  bitwise_and_1 = arg4_1 = full_1 = None
        view_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg5_1, _shape_param_4);  arg5_1 = _shape_param_4 = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg6_1, 2, 0, 128);  arg6_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view_1, full_2);  slice_2 = view_1 = full_2 = None
        add: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = where = None
        sub: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add, arg7_1);  add = arg7_1 = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg8_1);  exp = arg8_1 = None
        mul: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = div = None
        return mul

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
