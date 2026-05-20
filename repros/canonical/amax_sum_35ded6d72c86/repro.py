"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: 35ded6d72c86
Shape hash: 73112382
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 128], i64, max=128), T([512, 768], bf16), T([48, 128, 128], bf16), S([4, 128, 768]), S([4, -1, 12, 64]), S([4, 12, 128, 128]), S([4, 12, 128, 128]), S([48, 128, 128]), S([4, 12, 128, 64]), S([48, 128, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 128]", addmm_2: "bf16[512, 768]", bmm: "bf16[48, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        gt_scalar: "b8[4, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        unsqueeze_default: "b8[4, 1, 128]" = torch.ops.aten.unsqueeze.default(gt_scalar, 1);  gt_scalar = None
        repeat_default: "b8[4, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze_default, [1, 128, 1]);  unsqueeze_default = None
        unsqueeze_default_1: "b8[4, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat_default, 1);  repeat_default = None
        view_default: "bf16[4, 128, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None
        view_default_1: "bf16[4, 128, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "bf16[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "bf16[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_2);  bmm = _shape_param_2 = None
        div_tensor: "bf16[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_default_2, 8.0);  view_default_2 = None
        eq_scalar: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor);  eq_scalar = full_default = div_tensor = None
        convert_element_type_default: "f32[4, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        amax_default: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1: "bf16[4, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        expand_default: "bf16[4, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_default_1, _shape_param_3);  convert_element_type_default_1 = _shape_param_3 = None
        view_default_3: "bf16[48, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None
        expand_default_1: "bf16[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "bf16[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_4: "bf16[48, 128, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None
        return (view_default_3, view_default_4)


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
