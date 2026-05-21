"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train_001
Pattern hash: 0930a6da0722
Shape hash: 5a1fbf8b
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
_shapes_config = "(T([25216, 2304], f32), S([2304]))"

class Repro(torch.nn.Module):
    def forward(self, view_67: "f32[25216, 2304]", _shape_param_0):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_67, [0], True);  view_67 = None
        view_default: "f32[2304]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        slice_tensor: "f32[768]" = torch.ops.aten.slice.Tensor(view_default, 0, 0, 768)
        slice_tensor_1: "f32[768]" = torch.ops.aten.slice.Tensor(view_default, 0, 1536, 2304);  view_default = None
        return (slice_tensor_1, slice_tensor)



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
