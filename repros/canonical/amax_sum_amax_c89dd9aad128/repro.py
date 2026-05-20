"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-1-6-linux.aws.a100_graph22
Pattern hash: c89dd9aad128
Shape hash: 7244d22e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 2], bf16), T([512, 20005], bf16), S([4, 128, 20005]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_72: "bf16[4, 2]", addmm_73: "bf16[512, 20005]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[4, 2]" = torch.ops.prims.convert_element_type.default(addmm_72, torch.float32);  addmm_72 = None
        amax_default: "f32[4, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor: "f32[4, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[4, 2]" = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list: "f32[4, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True);  exp_default = None
        log_default: "f32[4, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[4, 2]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "bf16[4, 2]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        view_default: "bf16[4, 128, 20005]" = torch.ops.aten.view.default(addmm_73, _shape_param_0);  addmm_73 = _shape_param_0 = None
        convert_element_type_default_2: "f32[4, 128, 20005]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        amax_default_1: "f32[4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor_2: "f32[4, 128, 20005]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default_1);  convert_element_type_default_2 = amax_default_1 = None
        exp_default_1: "f32[4, 128, 20005]" = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list_1: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True);  exp_default_1 = None
        log_default_1: "f32[4, 128, 1]" = torch.ops.aten.log.default(sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3: "f32[4, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default_1);  sub_tensor_2 = log_default_1 = None
        convert_element_type_default_3: "bf16[4, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub_tensor_3, torch.bfloat16);  sub_tensor_3 = None
        return (convert_element_type_default_1, convert_element_type_default_3)


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
