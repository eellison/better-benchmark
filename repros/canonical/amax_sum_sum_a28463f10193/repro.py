"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s1_g5
Pattern hash: a28463f10193
Shape hash: a44dc58f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "bf16[1024, 50272]", arg197_1: "i64[1, 1024]"):
        # No stacktrace found for following nodes
        slice_tensor: "bf16[1024, 50265]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -7);  mm = None
        reshape_default: "bf16[1, 1024, 50265]" = torch.ops.aten.reshape.default(slice_tensor, [1, 1024, 50265]);  slice_tensor = None
        reshape_default_1: "bf16[1024, 50265]" = torch.ops.aten.reshape.default(reshape_default, [-1, 50265]);  reshape_default = None
        reshape_default_2: "i64[1024]" = torch.ops.aten.reshape.default(arg197_1, [-1]);  arg197_1 = None
        convert_element_type_default: "f32[1024, 50265]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        amax_default: "f32[1024, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        sub_tensor: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[1024, 50265]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[1024, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[1024, 50265]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "bf16[1024, 50265]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        ne_scalar: "b8[1024]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[1024]" = torch.ops.aten.where.self(ne_scalar, reshape_default_2, full_default);  ne_scalar = full_default = None
        unsqueeze_default: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "bf16[1024, 1]" = torch.ops.aten.gather.default(convert_element_type_default_1, 1, unsqueeze_default);  convert_element_type_default_1 = unsqueeze_default = None
        squeeze_dim: "bf16[1024]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "bf16[1024]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        ne_scalar_1: "b8[1024]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100)
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "bf16[1024]" = torch.ops.aten.where.self(ne_scalar_1, neg_default, full_default_1);  ne_scalar_1 = neg_default = full_default_1 = None
        ne_scalar_2: "b8[1024]" = torch.ops.aten.ne.Scalar(reshape_default_2, -100);  reshape_default_2 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar_2);  ne_scalar_2 = None
        convert_element_type_default_2: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.bfloat16);  sum_default = None
        sum_default_1: "bf16[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "bf16[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_2);  sum_default_1 = convert_element_type_default_2 = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([1024, 50272], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [1, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
