"""
Standalone repro captured via capture_hook.
Label: maml
Pattern hash: fba82396802b
Shape hash: 49ab4aac
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[5, 64, 1, 1]", arg1_1: "bf16[64]", arg2_1: "bf16[64]", arg3_1: "i64[5]", arg4_1: "bf16[64]", arg5_1: "bf16[64]", arg6_1: "bf16[5, 64]", arg7_1: "bf16[5]", _shape_param_0):
        # No stacktrace found for following nodes
        relu: "bf16[5, 64, 1, 1]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        convert_element_type: "f32[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(relu, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type = None
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_1: "bf16[64]" = torch.ops.aten.mul.Tensor(arg1_1, 0.9)
        add: "f32[64]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
        convert_element_type_1: "bf16[64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3])
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, 1.25);  squeeze_1 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_2, 0.1);  mul_2 = None
        mul_4: "bf16[64]" = torch.ops.aten.mul.Tensor(arg2_1, 0.9)
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        convert_element_type_2: "bf16[64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        ne: "b8[5]" = torch.ops.aten.ne.Scalar(arg3_1, -100)
        sub: "f32[5, 64, 1, 1]" = torch.ops.aten.sub.Tensor(relu, getitem_1);  relu = getitem_1 = None
        add_2: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_5: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        unsqueeze: "bf16[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_1: "bf16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[5, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_5, unsqueeze_1);  mul_5 = unsqueeze_1 = None
        unsqueeze_2: "bf16[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_3: "bf16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_3: "f32[5, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[5, 64, 1, 1]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[5, 64]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_0);  convert_element_type_3 = _shape_param_0 = None
        permute: "bf16[64, 5]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[5, 5]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        convert_element_type_4: "f32[5, 5]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        amax: "f32[5, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub_1: "f32[5, 5]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[5, 5]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[5, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[5, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[5, 5]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        convert_element_type_5: "bf16[5, 5]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        ne_1: "b8[5]" = torch.ops.aten.ne.Scalar(arg3_1, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[5]" = torch.ops.aten.where.self(ne_1, arg3_1, full);  ne_1 = full = None
        unsqueeze_4: "i64[5, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "bf16[5, 1]" = torch.ops.aten.gather.default(convert_element_type_5, 1, unsqueeze_4);  convert_element_type_5 = unsqueeze_4 = None
        squeeze_2: "bf16[5]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[5]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[5]" = torch.ops.aten.where.self(ne, neg, full_1);  ne = neg = full_1 = None
        sum_2: "bf16[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        ne_2: "b8[5]" = torch.ops.aten.ne.Scalar(arg3_1, -100);  arg3_1 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_6: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.bfloat16);  sum_3 = None
        div: "bf16[]" = torch.ops.aten.div.Tensor(sum_2, convert_element_type_6);  sum_2 = convert_element_type_6 = None
        copy_: "bf16[64]" = torch.ops.aten.copy_.default(arg1_1, convert_element_type_1);  arg1_1 = convert_element_type_1 = None
        copy__1: "bf16[64]" = torch.ops.aten.copy_.default(arg2_1, convert_element_type_2);  arg2_1 = convert_element_type_2 = None
        return (addmm, div, copy_, copy__1)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
