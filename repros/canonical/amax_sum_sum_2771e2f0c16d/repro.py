"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph4
Pattern hash: 2771e2f0c16d
Shape hash: 7062ba34
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64, max=512), T([512, 32000], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([1, 512, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg113_1: "i64[1, 512]", addmm_26: "f32[512, 32000]", rsqrt_24: "f32[1, 512, 1]", rsqrt_23: "f32[1, 512, 1]", rsqrt_22: "f32[1, 512, 1]", rsqrt_21: "f32[1, 512, 1]", rsqrt_20: "f32[1, 512, 1]", rsqrt_19: "f32[1, 512, 1]", rsqrt_18: "f32[1, 512, 1]", rsqrt_17: "f32[1, 512, 1]", rsqrt_16: "f32[1, 512, 1]", rsqrt_15: "f32[1, 512, 1]", rsqrt_14: "f32[1, 512, 1]", rsqrt_13: "f32[1, 512, 1]", rsqrt_12: "f32[1, 512, 1]", rsqrt_11: "f32[1, 512, 1]", rsqrt_10: "f32[1, 512, 1]", rsqrt_9: "f32[1, 512, 1]", rsqrt_8: "f32[1, 512, 1]", rsqrt_7: "f32[1, 512, 1]", rsqrt_6: "f32[1, 512, 1]", rsqrt_5: "f32[1, 512, 1]", rsqrt_4: "f32[1, 512, 1]", rsqrt_3: "f32[1, 512, 1]", rsqrt_2: "f32[1, 512, 1]", rsqrt_1: "f32[1, 512, 1]", rsqrt: "f32[1, 512, 1]"):
        # No stacktrace found for following nodes
        view_default: "i64[512]" = torch.ops.aten.view.default(arg113_1, [-1]);  arg113_1 = None
        amax_default: "f32[512, 1]" = torch.ops.aten.amax.default(addmm_26, [1], True)
        sub_tensor: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(addmm_26, amax_default);  addmm_26 = amax_default = None
        exp_default: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[512, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        ne_scalar: "b8[512]" = torch.ops.aten.ne.Scalar(view_default, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[512]" = torch.ops.aten.where.self(ne_scalar, view_default, full_default);  view_default = full_default = None
        unsqueeze_default: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "f32[512, 1]" = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim: "f32[512]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "f32[512]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[512]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default: "f32[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.float32);  sum_default = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "f32[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default);  sum_default_1 = convert_element_type_default = None
        div_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        div_tensor_2: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        div_tensor_3: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        div_tensor_4: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        div_tensor_5: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        div_tensor_6: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        div_tensor_7: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        div_tensor_8: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        div_tensor_9: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        div_tensor_10: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        div_tensor_11: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        div_tensor_12: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        div_tensor_13: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_tensor_14: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        div_tensor_15: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_tensor_16: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        div_tensor_17: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_tensor_18: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        div_tensor_19: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_tensor_20: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        div_tensor_21: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_tensor_22: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        div_tensor_23: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        div_tensor_24: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        div_tensor_25: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5, div_tensor_6, div_tensor_7, div_tensor_8, div_tensor_9, div_tensor_10, div_tensor_11, div_tensor_12, div_tensor_13, div_tensor_14, div_tensor_15, div_tensor_16, div_tensor_17, div_tensor_18, div_tensor_19, div_tensor_20, div_tensor_21, div_tensor_22, div_tensor_23, div_tensor_24, div_tensor_25)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
