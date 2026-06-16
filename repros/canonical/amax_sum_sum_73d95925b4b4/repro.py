"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_train
Pattern hash: 73d95925b4b4
Shape hash: fe37f3bb
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
    def forward(self, arg0_1: "bf16[128, 8]", arg1_1: "i64[1]", arg2_1: "i64[1]", arg3_1: "f32[]", _shape_param_0):
        # No stacktrace found for following nodes
        slice_1: "bf16[128, 2]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -6);  arg0_1 = None
        view: "bf16[1, 128, 2]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        split = torch.ops.aten.split.Tensor(view, 1, -1);  view = None
        getitem: "bf16[1, 128, 1]" = split[0]
        getitem_1: "bf16[1, 128, 1]" = split[1];  split = None
        squeeze: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
        clone: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
        clone_1: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg1_1, 0);  arg1_1 = None
        clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
        clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg2_1, 0);  arg2_1 = None
        clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
        convert_element_type: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone, torch.float32)
        amax: "f32[1, 1]" = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = None
        exp: "f32[1, 128]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[1, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub, log);  sub = None
        convert_element_type_1: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_2: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[1]" = torch.ops.aten.where.self(ne, clamp_max, full);  clamp_max = None
        unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[1, 1]" = torch.ops.aten.gather.default(convert_element_type_2, 1, unsqueeze);  convert_element_type_2 = unsqueeze = None
        squeeze_2: "f32[1]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[1]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_1: "f32[1]" = torch.ops.aten.where.self(ne, neg, arg3_1);  neg = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne)
        convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_3);  sum_3 = convert_element_type_3 = None
        convert_element_type_4: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32)
        amax_1: "f32[1, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax_1);  convert_element_type_4 = None
        exp_1: "f32[1, 128]" = torch.ops.aten.exp.default(sub_2)
        sum_4: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_2, log_1);  sub_2 = None
        convert_element_type_5: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_3, torch.bfloat16);  sub_3 = None
        convert_element_type_6: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        where_2: "i64[1]" = torch.ops.aten.where.self(ne_1, clamp_max_1, full);  clamp_max_1 = full = None
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1: "f32[1, 1]" = torch.ops.aten.gather.default(convert_element_type_6, 1, unsqueeze_1);  convert_element_type_6 = unsqueeze_1 = None
        squeeze_3: "f32[1]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_3: "f32[1]" = torch.ops.aten.where.self(ne_1, neg_1, arg3_1);  neg_1 = arg3_1 = None
        sum_5: "i64[]" = torch.ops.aten.sum.default(ne_1)
        convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_6, convert_element_type_7);  sum_6 = convert_element_type_7 = None
        add: "f32[]" = torch.ops.aten.add.Tensor(div, div_1);  div = div_1 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(add, 2);  add = None
        return (clone, clone_1, amax, log, ne, amax_1, log_1, ne_1, div_2)



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
