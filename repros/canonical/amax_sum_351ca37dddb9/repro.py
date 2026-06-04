"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer_000
Pattern hash: 351ca37dddb9
Shape hash: 195b1db9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 32, 33], f32), S([32, 8, 32, 33]), S([32, 8, 32, 33]), S([256, 32, 33]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_14: "f32[256, 32, 33]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 8, 32, 33]" = torch.ops.aten.view.default(bmm_14, _shape_param_0);  bmm_14 = _shape_param_0 = None
        mul_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.mul.Tensor(view_default, 1);  view_default = None
        amax_default: "f32[32, 8, 32, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        div_tensor: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[32, 8, 32, 33]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_dim_int_list: "f32[32, 8, 32, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[32, 8, 32, 33]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        expand_default: "f32[32, 8, 32, 33]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_1);  div_tensor_1 = _shape_param_1 = None
        view_default_1: "f32[256, 32, 33]" = torch.ops.aten.view.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None
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
