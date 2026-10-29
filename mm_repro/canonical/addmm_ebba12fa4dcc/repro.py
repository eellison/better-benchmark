"""Curated MM repro. Pattern hash: ebba12fa4dcc."""
import sys
from pathlib import Path
from math import inf, nan
import torch
import torch._inductor.inductor_prims
from torch import device
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, load_shape_configs, make_inputs_from_config
_repro_version = 3

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[5, 64, 1, 1]", arg1_1: "i64[5]", arg2_1: "bf16[64]", arg3_1: "bf16[64]", arg4_1: "bf16[64, 5]", arg5_1: "bf16[5]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        relu: "bf16[5, 64, 1, 1]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        convert_element_type: "f32[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(relu, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        ne: "b8[5]" = torch.ops.aten.ne.Scalar(arg1_1, -100)
        sub: "f32[5, 64, 1, 1]" = torch.ops.aten.sub.Tensor(relu, getitem_1);  relu = getitem_1 = None
        add: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze_1: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        mul_1: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_3: "bf16[64, 1, 1]" = torch.ops.aten.view.default(arg3_1, _shape_param_1);  arg3_1 = _shape_param_1 = None
        add_1: "f32[5, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_3);  mul_1 = unsqueeze_3 = None
        convert_element_type_1: "bf16[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[5, 64]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg5_1, view, arg4_1);  arg5_1 = view = arg4_1 = None
        convert_element_type_2: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub_1: "f32[5, 5]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        convert_element_type_3: "bf16[5, 5]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[5]" = torch.ops.aten.where.self(ne, arg1_1, full);  arg1_1 = full = None
        unsqueeze_4: "i64[5, 1]" = torch.ops.aten.view.default(where, _shape_param_3);  where = _shape_param_3 = None
        gather: "bf16[5, 1]" = torch.ops.aten.gather.default(convert_element_type_3, 1, unsqueeze_4);  convert_element_type_3 = unsqueeze_4 = None
        squeeze: "bf16[5]" = torch.ops.aten.view.default(gather, _shape_param_4);  gather = _shape_param_4 = None
        neg: "bf16[5]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[5]" = torch.ops.aten.where.self(ne, neg, full_1);  neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_4: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_4);  sum_2 = convert_element_type_4 = None
        return (addmm, div)

def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = next(iter(load_shape_configs(__file__).values()))
    return make_inputs_from_config(shape_config)

if __name__ == "__main__":
    benchmark_repro(Repro(), make_inputs, __file__)
