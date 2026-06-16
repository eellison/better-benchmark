"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: e7c5599e5aa2
Shape hash: 89a27d2d
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
    def forward(self, arg0_1: "bf16[512, 4, 256, 40]", arg1_1: "bf16[512, 4, 256, 40]", arg2_1: "bf16[512, 4, 256, 40]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        constant_pad_nd: "bf16[512, 4, 256, 36]" = torch.ops.aten.constant_pad_nd.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd_1: "bf16[512, 4, 256, 36]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        constant_pad_nd_2: "bf16[512, 4, 256, 36]" = torch.ops.aten.constant_pad_nd.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "bf16[1536, 4, 256, 36]" = torch.ops.aten.cat.default([constant_pad_nd_2, constant_pad_nd_1, constant_pad_nd]);  constant_pad_nd_2 = constant_pad_nd_1 = constant_pad_nd = None
        view: "bf16[3, 512, 4, 256, 36]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute: "bf16[512, 256, 3, 4, 36]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[512, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[512, 256, 432]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_2: "bf16[131072, 432]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        permute_1: "bf16[432, 131072]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[432]" = torch.ops.aten.view.default(sum_1, _shape_param_6);  sum_1 = _shape_param_6 = None
        convert_element_type: "bf16[432]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_1: "f32[432]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_2, permute_1, convert_element_type_1)



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
