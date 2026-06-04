"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_003
Pattern hash: 13195092a57b
Shape hash: 5af27dfa
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 1500, 384], f32), S([12000, 384]), S([384]))"

class Repro(torch.nn.Module):
    def forward(self, arg25_1: "f32[8, 1500, 384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(arg25_1, memory_format = torch.contiguous_format);  arg25_1 = None
        view_default: "f32[12000, 384]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default: "f32[384, 12000]" = torch.ops.aten.permute.default(view_default, [1, 0])
        sum_dim_int_list: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_default, [0], True);  view_default = None
        view_default_1: "f32[384]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_1);  sum_dim_int_list = _shape_param_1 = None
        return (permute_default, view_default_1)

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
