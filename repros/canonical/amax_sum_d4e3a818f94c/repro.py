"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer_000
Pattern hash: d4e3a818f94c
Shape hash: 8b08115c
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
_shapes_config = "(T([128, 128], i64), T([1536, 128, 128], f32), S([128, 12, 128, 128]), S([128, 12, 128, 128]), S([1536, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[128, 128]", bmm: "f32[1536, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        gt_scalar: "b8[128, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze_default: "b8[128, 1, 128]" = torch.ops.aten.unsqueeze.default(gt_scalar, 1);  gt_scalar = None
        repeat_default: "b8[128, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze_default, [1, 128, 1]);  unsqueeze_default = None
        unsqueeze_default_1: "b8[128, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat_default, 1);  repeat_default = None
        eq_scalar: "b8[128, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], -1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[128, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        div_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_default, 8.0);  view_default = None
        where_self: "f32[128, 12, 128, 128]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor);  eq_scalar = full_default = div_tensor = None
        amax_default: "f32[128, 12, 128, 1]" = torch.ops.aten.amax.default(where_self, [-1], True)
        sub_tensor: "f32[128, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_self, amax_default);  where_self = amax_default = None
        exp_default: "f32[128, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[128, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[128, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_1);  div_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[1536, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
        return view_default_1



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
