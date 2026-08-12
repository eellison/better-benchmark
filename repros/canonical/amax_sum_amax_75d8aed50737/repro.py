"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 75d8aed50737
Shape hash: 5d077752
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
    def forward(self, arg0_1: "bf16[64, 1024, 1024]", arg1_1: "f32[32, 8]", arg2_1: "i64[64]", arg3_1: "bf16[64, 1024, 1024]", arg4_1: "f32[32, 8]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        full: "f32[8, 1, 1024, 1024]" = torch.ops.aten.full.default(_shape_param_0, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        view: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_1);  arg0_1 = _shape_param_1 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        add: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze, 0);  unsqueeze = None
        unsqueeze_1: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0)
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_1, add);  unsqueeze_1 = add = None
        gt: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
        add_1: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub)
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_1: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        convert_element_type_2: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
        full_1: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_2, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_1);  add_2 = full_1 = None
        where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
        embedding: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg1_1, add_3);  arg1_1 = None
        permute: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding, [2, 0, 1])
        unsqueeze_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute, 0);  permute = None
        add_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_2, full);  unsqueeze_2 = full = None
        add_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view, add_4);  view = None
        convert_element_type_3: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        convert_element_type_4: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_5: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 1)
        inductor_random: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_6: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt_1: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_6, 0.1);  convert_element_type_6 = None
        mul_2: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_1, convert_element_type_5);  convert_element_type_5 = None
        mul_3: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        expand: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_3, _shape_param_4);  mul_3 = _shape_param_4 = None
        view_1: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand, _shape_param_5);  expand = _shape_param_5 = None
        add_6: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_6, 0);  add_6 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
        unsqueeze_6: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2)
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_5);  unsqueeze_6 = unsqueeze_5 = None
        expand_1: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_6);  le = _shape_param_6 = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_1, full_2, full_3);  expand_1 = full_3 = None
        view_2: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_7);  arg3_1 = _shape_param_7 = None
        full_4: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_8, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_4);  sub = full_4 = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_7: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_3: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_7, 16);  convert_element_type_7 = None
        log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_3);  div_3 = None
        div_4: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_4: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_4, 16);  div_4 = None
        convert_element_type_8: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
        add_7: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_8, 16);  convert_element_type_8 = None
        full_5: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_9, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_7, full_5);  add_7 = full_5 = None
        where_2: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_8: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_2, 0);  where_2 = None
        embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg4_1, add_8);  arg4_1 = None
        permute_1: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_1, 0);  permute_1 = None
        add_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_7, where_1);  unsqueeze_7 = where_1 = None
        add_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_2, add_9);  view_2 = None
        convert_element_type_9: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        convert_element_type_10: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        amax_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = None
        exp_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_11: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg2_1, 27);  arg2_1 = None
        inductor_random_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_10, inductor_lookup_seed_1, 'rand');  _shape_param_10 = inductor_lookup_seed_1 = None
        convert_element_type_12: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_2: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_12, 0.1);  convert_element_type_12 = None
        mul_5: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_2, convert_element_type_11);  convert_element_type_11 = None
        mul_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        expand_2: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_6, _shape_param_11);  mul_6 = _shape_param_11 = None
        view_3: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_2, _shape_param_12);  expand_2 = _shape_param_12 = None
        permute_2: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_3, [0, 2, 1])
        permute_3: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (add_3, embedding, add_4, amax, sum_1, gt_1, view_1, unsqueeze_4, full_2, add_8, embedding_1, add_9, amax_1, sum_2, gt_2, view_3, permute_2, permute_3)



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
