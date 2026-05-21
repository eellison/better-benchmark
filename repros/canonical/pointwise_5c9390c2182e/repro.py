"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer_000
Pattern hash: 5c9390c2182e
Shape hash: 0c2e0a69
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
_shapes_config = "(T([64, 1024, 8, 64], f32), T([1024, 512], f32), S([32, 32, 512]), S([32, 32, 8, 64]), S([32, 8, 33, 64]), S([256, 33, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg85_1: "f32[64, 1024, 8, 64]", mm_51: "f32[1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        slice_tensor: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_tensor_1: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(arg85_1, 0, 0, 32)
        slice_tensor_2: "f32[32, 32, 8, 64]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 1, 33);  slice_tensor_1 = None
        view_default: "f32[32, 32, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_0);  mm_51 = _shape_param_0 = None
        view_default_1: "f32[32, 32, 8, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        copy_default: "f32[32, 32, 8, 64]" = torch.ops.aten.copy.default(slice_tensor_2, view_default_1);  slice_tensor_2 = view_default_1 = None
        slice_scatter_default: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default, 1, 1, 33);  slice_tensor = copy_default = None
        slice_scatter_default_1: "f32[64, 1024, 8, 64]" = torch.ops.aten.slice_scatter.default(arg85_1, slice_scatter_default, 0, 0, 32);  slice_scatter_default = None
        slice_tensor_3: "f32[32, 1024, 8, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 0, 32)
        slice_tensor_4: "f32[32, 33, 8, 64]" = torch.ops.aten.slice.Tensor(slice_tensor_3, 1, 0, 33);  slice_tensor_3 = None
        permute_default: "f32[32, 8, 33, 64]" = torch.ops.aten.permute.default(slice_tensor_4, [0, 2, 1, 3]);  slice_tensor_4 = None
        expand_default: "f32[32, 8, 33, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default: "f32[32, 8, 33, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_2: "f32[256, 33, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        copy__default: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg85_1, slice_scatter_default_1);  arg85_1 = slice_scatter_default_1 = None
        return (view_default_2, copy__default)



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
