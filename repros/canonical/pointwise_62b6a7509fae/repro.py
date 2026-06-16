"""
Standalone repro captured via capture_hook.
Label: timm_beit_base_patch16_224_train
Pattern hash: 62b6a7509fae
Shape hash: 0b512413
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[197, 197]", arg1_1: "f32[732, 12]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "i64[38809]" = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        index: "f32[38809, 12]" = torch.ops.aten.index.Tensor(arg1_1, [view]);  arg1_1 = view = None
        view_1: "f32[197, 197, 12]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
        permute: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_1, [2, 0, 1]);  view_1 = None
        clone: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        unsqueeze: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone, 0);  clone = None
        convert_element_type: "bf16[1, 12, 197, 197]" = torch.ops.prims.convert_element_type.default(unsqueeze, torch.bfloat16);  unsqueeze = None
        constant_pad_nd: "bf16[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(convert_element_type, _shape_param_1, 0.0);  convert_element_type = _shape_param_1 = None
        slice_1: "bf16[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 197)
        expand: "bf16[128, 12, 197, 197]" = torch.ops.aten.expand.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        return (constant_pad_nd, expand)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
