"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 239899b9fcf3
Shape hash: b08406d1
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_124: "f64[64, 1024, 64]", bmm_125: "f64[64, 1024, 64]", getitem_221: "f64[16, 4096, 80]", bmm_126: "f64[16, 4096, 4096]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        view_default: "f64[64, 16, 64, 1, 64]" = torch.ops.aten.view.default(bmm_124, _shape_param_0);  bmm_124 = _shape_param_0 = None
        permute_default: "f64[16, 64, 64, 64, 1]" = torch.ops.aten.permute.default(view_default, [1, 0, 2, 4, 3]);  view_default = None
        view_default_1: "f64[16, 64, 64, 64]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        view_default_2: "f64[64, 16, 64, 1, 64]" = torch.ops.aten.view.default(bmm_125, _shape_param_2);  bmm_125 = _shape_param_2 = None
        permute_default_1: "f64[16, 64, 64, 64, 1]" = torch.ops.aten.permute.default(view_default_2, [1, 2, 0, 4, 3]);  view_default_2 = None
        view_default_3: "f64[16, 64, 64, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_3);  permute_default_1 = _shape_param_3 = None
        unsqueeze_default: "f64[16, 64, 64, 64, 1]" = torch.ops.aten.unsqueeze.default(view_default_1, -1);  view_default_1 = None
        unsqueeze_default_1: "f64[16, 64, 64, 1, 64]" = torch.ops.aten.unsqueeze.default(view_default_3, -2);  view_default_3 = None
        clone_default: "f64[16, 64, 64, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        view_default_4: "f64[16, 4096, 64, 1]" = torch.ops.aten.view.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        clone_default_1: "f64[16, 64, 64, 1, 64]" = torch.ops.aten.clone.default(unsqueeze_default_1, memory_format = torch.contiguous_format);  unsqueeze_default_1 = None
        view_default_5: "f64[16, 4096, 1, 64]" = torch.ops.aten.view.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        view_default_6: "f64[1, 16, 4096, 80]" = torch.ops.aten.view.default(getitem_221, _shape_param_6);  getitem_221 = _shape_param_6 = None
        view_default_7: "f64[1, 16, 4096, 64, 1]" = torch.ops.aten.view.default(view_default_4, _shape_param_7);  view_default_4 = _shape_param_7 = None
        view_default_8: "f64[1, 16, 4096, 1, 64]" = torch.ops.aten.view.default(view_default_5, _shape_param_8);  view_default_5 = _shape_param_8 = None
        add_tensor: "f64[1, 16, 4096, 64, 64]" = torch.ops.aten.add.Tensor(view_default_7, view_default_8);  view_default_7 = view_default_8 = None
        view_default_9: "f64[1, 16, 4096, 4096]" = torch.ops.aten.view.default(add_tensor, _shape_param_9);  add_tensor = _shape_param_9 = None
        view_default_10: "f64[1, 16, 4096, 4096]" = torch.ops.aten.view.default(bmm_126, _shape_param_10);  bmm_126 = _shape_param_10 = None
        add_tensor_1: "f64[1, 16, 4096, 4096]" = torch.ops.aten.add.Tensor(view_default_10, view_default_9);  view_default_10 = view_default_9 = None
        amax_default: "f64[1, 16, 4096, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor: "f64[1, 16, 4096, 4096]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default);  amax_default = None
        exp_default: "f64[1, 16, 4096, 4096]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f64[1, 16, 4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f64[1, 16, 4096, 4096]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        eq_scalar: "b8[1, 16, 4096, 4096]" = torch.ops.aten.eq.Scalar(add_tensor_1, -inf);  add_tensor_1 = None
        logical_not_default: "b8[1, 16, 4096, 4096]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[1, 16, 4096, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[1, 16, 4096, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f64[1, 16, 4096, 4096]" = torch.ops.aten.full.default([1, 16, 4096, 4096], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f64[1, 16, 4096, 4096]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default: "f64[1, 16, 4096, 4096]" = torch.ops.aten.expand.default(where_self, _shape_param_11);  where_self = _shape_param_11 = None
        view_default_11: "f64[16, 4096, 4096]" = torch.ops.aten.view.default(expand_default, _shape_param_12);  expand_default = _shape_param_12 = None
        expand_default_1: "f64[1, 16, 4096, 80]" = torch.ops.aten.expand.default(view_default_6, _shape_param_13);  view_default_6 = _shape_param_13 = None
        view_default_12: "f64[16, 4096, 80]" = torch.ops.aten.view.default(expand_default_1, _shape_param_14);  expand_default_1 = _shape_param_14 = None
        return (view_default_11, view_default_12)


def _default_make_inputs():
    return [
    torch.randn([64, 1024, 64], dtype=torch.float64, device='cuda'),
    torch.randn([64, 1024, 64], dtype=torch.float64, device='cuda'),
    torch.randn(15726080, dtype=torch.float64, device='cuda').as_strided([16, 4096, 80], [80, 3840, 1]),  # getitem_221
    torch.randn([16, 4096, 4096], dtype=torch.float64, device='cuda'),
    [64, 16, 64, 1, 64],  # _shape_param_0
    [16, 64, 64, 64],  # _shape_param_1
    [64, 16, 64, 1, 64],  # _shape_param_2
    [16, 64, 64, 64],  # _shape_param_3
    [16, 4096, 64, 1],  # _shape_param_4
    [16, 4096, 1, 64],  # _shape_param_5
    [1, 16, 4096, -1],  # _shape_param_6
    [1, 16, 4096, 64, 1],  # _shape_param_7
    [1, 16, 4096, 1, 64],  # _shape_param_8
    [1, 16, 4096, 4096],  # _shape_param_9
    [1, 16, 4096, 4096],  # _shape_param_10
    [1, 16, 4096, 4096],  # _shape_param_11
    [16, 4096, 4096],  # _shape_param_12
    [1, 16, 4096, 80],  # _shape_param_13
    [16, 4096, 80],  # _shape_param_14
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
