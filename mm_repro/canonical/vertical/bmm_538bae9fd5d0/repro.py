"""Curated MM repro. Pattern hash: 538bae9fd5d0."""
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
    def forward(self, arg0_1: "bf16[192, 128, 64]", arg1_1: "bf16[192, 64, 128]", arg2_1: "b8[32, 6, 128, 128]", arg3_1: "bf16[192, 128, 128]", arg4_1: "f32[6, 128, 128]", arg5_1: "f32[32, 6, 128, 1]", arg6_1: "f32[32, 6, 128, 1]", arg7_1: "f32[32, 6, 128, 128]", arg8_1: "bf16[192, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        convert_element_type: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        mul: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_1: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view, mul);  view = mul = None
        convert_element_type_1: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        full: "f32[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_1, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        view_1: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        unsqueeze: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        add: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze, full);  unsqueeze = full = None
        add_1: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_1, add);  view_1 = add = None
        convert_element_type_2: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_3: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        sub: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg5_1);  convert_element_type_3 = arg5_1 = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, arg6_1);  exp = arg6_1 = None
        mul_2: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1, div);  convert_element_type_1 = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [3], True)
        neg: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul_2);  neg = sum_1 = mul_2 = None
        convert_element_type_4: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_4: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  _shape_param_3 = None
        convert_element_type_5: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        add_2: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(arg7_1, convert_element_type_5);  arg7_1 = convert_element_type_5 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_4, arg8_1);  arg8_1 = None
        view_5: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_4);  bmm_1 = _shape_param_4 = None
        permute_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_7: "bf16[4096, 384]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        permute_2: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_7, [1, 0])
        return (view_4, add_2, view_7, permute_2)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
