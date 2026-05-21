"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_005
Pattern hash: d112f48ea917
Shape hash: ac7e5877
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 12, 4096], i64), T([6144, 64, 128], f32), T([], f32), S([8, 12, -1, 64]), S([8, 12, 64, 64, 128]), S([8, 12, 64, 64, 128]), S([6144, 64, 128]))"

class Repro(torch.nn.Module):
    def forward(self, remainder_1: "i64[8, 12, 4096]", bmm_1: "f32[6144, 64, 128]", arg5_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "i64[8, 12, 64, 64]" = torch.ops.aten.view.default(remainder_1, _shape_param_0);  remainder_1 = _shape_param_0 = None
        unsqueeze_default: "i64[8, 12, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default, -1)
        slice_tensor: "i64[8, 12, 1, 64]" = torch.ops.aten.slice.Tensor(view_default, 2, -1, 9223372036854775807)
        slice_tensor_1: "i64[8, 12, 63, 64]" = torch.ops.aten.slice.Tensor(view_default, 2, 0, -1)
        cat_default: "i64[8, 12, 64, 64]" = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2);  slice_tensor = slice_tensor_1 = None
        cat_default_1: "i64[8, 12, 64, 128]" = torch.ops.aten.cat.default([cat_default, view_default], 3);  cat_default = view_default = None
        unsqueeze_default_1: "i64[8, 12, 64, 1, 128]" = torch.ops.aten.unsqueeze.default(cat_default_1, -2);  cat_default_1 = None
        ne_tensor: "b8[8, 12, 64, 64, 128]" = torch.ops.aten.ne.Tensor(unsqueeze_default, unsqueeze_default_1);  unsqueeze_default = unsqueeze_default_1 = None
        view_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_1);  bmm_1 = _shape_param_1 = None
        where_self: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.where.self(ne_tensor, view_default_1, arg5_1);  ne_tensor = view_default_1 = arg5_1 = None
        amax_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        abs_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.abs.default(amax_default)
        eq_scalar: "b8[8, 12, 64, 64, 1]" = torch.ops.aten.eq.Scalar(abs_default, inf);  abs_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.where.self(eq_scalar, full_default, amax_default);  eq_scalar = full_default = amax_default = None
        sub_tensor: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, where_self_1)
        exp_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        add_tensor: "f32[8, 12, 64, 64, 1]" = torch.ops.aten.add.Tensor(log_default, where_self_1);  log_default = where_self_1 = None
        sub_tensor_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.sub.Tensor(where_self, add_tensor);  where_self = add_tensor = None
        exp_default_1: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        expand_default: "f32[8, 12, 64, 64, 128]" = torch.ops.aten.expand.default(exp_default_1, _shape_param_2);  exp_default_1 = _shape_param_2 = None
        view_default_2: "f32[6144, 64, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_3);  expand_default = _shape_param_3 = None
        return view_default_2



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
