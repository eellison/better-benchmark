"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_infer
Pattern hash: 57fd7ff76261
Shape hash: ea4c0a34
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
    def forward(self, arg0_1: "bf16[64, 1024, 1024]", arg1_1: "bf16[32, 8]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        add: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_1, 0);  unsqueeze_1 = None
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze, add);  unsqueeze = add = None
        full: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full);  sub = full = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum);  minimum = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type, 16);  convert_element_type = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.0794415416798357);  log = None
        mul: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 16);  div_1 = None
        convert_element_type_1: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_1, 16);  convert_element_type_1 = None
        full_1: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_2, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_1, full_1);  add_1 = full_1 = None
        where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, neg, minimum_1);  lt = neg = minimum_1 = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where, 0);  where = None
        embedding: "bf16[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg1_1, add_2);  arg1_1 = add_2 = None
        permute: "bf16[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding, [2, 0, 1]);  embedding = None
        unsqueeze_2: "bf16[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[1024]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None
        unsqueeze_6: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_7: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_8);  unsqueeze_5 = unsqueeze_8 = None
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_3);  le = _shape_param_3 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_2, full_3);  expand = full_2 = full_3 = None
        add_5: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_2, where_1);  unsqueeze_2 = where_1 = None
        add_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view, add_5);  view = None
        convert_element_type_2: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_1: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_3, _shape_param_4);  convert_element_type_3 = _shape_param_4 = None
        view_1: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_1, _shape_param_5);  expand_1 = _shape_param_5 = None
        return (add_5, view_1)



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
