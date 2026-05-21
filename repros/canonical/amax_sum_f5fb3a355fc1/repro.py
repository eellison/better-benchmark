"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_000
Pattern hash: f5fb3a355fc1
Shape hash: 82d559dd
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
_shapes_config = "(T([512, 128, 128], f32), T([1, 1, 2048, 2048], b8), T([], f32), T([32, 1, 128, 128], f32), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_46: "f32[512, 128, 128]", arg330_1: "b8[1, 1, 2048, 2048]", full_2: "f32[]", where: "f32[32, 1, 128, 128]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_0);  bmm_46 = _shape_param_0 = None
        slice_tensor: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg330_1, 2, 0, 128);  arg330_1 = None
        slice_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128);  slice_tensor = None
        where_self: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_tensor_1, view_default, full_2);  slice_tensor_1 = view_default = full_2 = None
        add_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_self, where);  where_self = where = None
        amax_default: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_tensor, _shape_param_1);  div_tensor = _shape_param_1 = None
        view_default_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
