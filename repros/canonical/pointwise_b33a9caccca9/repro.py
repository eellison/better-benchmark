"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_infer_000
Pattern hash: b33a9caccca9
Shape hash: 4c5ebe7f
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
_shapes_config = "(T([197, 197], i64, gen=Index(732)), T([732, 12], f32), S([197, 197, -1]), S([128, 12, 197, 197]))"

class Repro(torch.nn.Module):
    def forward(self, arg210_1: "i64[197, 197]", arg209_1: "f32[732, 12]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "i64[38809]" = torch.ops.aten.view.default(arg210_1, [-1]);  arg210_1 = None
        index_tensor: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg209_1, [view_default]);  arg209_1 = view_default = None
        view_default_1: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        permute_default: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_default_1, [2, 0, 1]);  view_default_1 = None
        clone_default: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None
        constant_pad_nd_default: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 3], 0.0);  unsqueeze_default = None
        slice_tensor: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_default, -1, 0, 197);  constant_pad_nd_default = None
        expand_default: "f32[128, 12, 197, 197]" = torch.ops.aten.expand.default(slice_tensor, _shape_param_1);  slice_tensor = _shape_param_1 = None
        return expand_default



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
