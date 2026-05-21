"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_007
Pattern hash: 34445852895e
Shape hash: 8ca1b46a
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
_shapes_config = "(T([8, 16, 1024, 64], f32, stride=(1048576, 64, 1024, 1)), S([8, 1024, 1024]), S([8192, 1024]), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[8, 16, 1024, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute_default: "f32[8, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default: "f32[8, 1024, 1024]" = torch.ops.aten.view.default(permute_default, _shape_param_0);  permute_default = _shape_param_0 = None
        view_default_1: "f32[8192, 1024]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default_1: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_default_1, [1, 0])
        sum_dim_int_list: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_default_1, [0], True);  view_default_1 = None
        view_default_2: "f32[1024]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_2);  sum_dim_int_list = _shape_param_2 = None
        return (permute_default_1, view_default_2)



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
