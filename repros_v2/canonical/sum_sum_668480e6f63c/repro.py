"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 668480e6f63c
Shape hash: 28a9d256
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
    def forward(self, arg0_1: "bf16[256, 512, 64]", arg1_1: "bf16[256, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        permute: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3]);  view = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute, [2, 0, 1, 4, 3]);  permute = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_1, 4);  permute_1 = None
        convert_element_type: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze, torch.float32);  squeeze = None
        convert_element_type_1: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16)
        sum_1: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type, [0, 1], True, dtype = torch.float32);  convert_element_type = None
        view_1: "f32[16, 64]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        view_2: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(arg1_1, _shape_param_2);  arg1_1 = _shape_param_2 = None
        permute_2: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        permute_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_2, [2, 0, 1, 4, 3]);  permute_2 = None
        squeeze_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_3, 4);  permute_3 = None
        convert_element_type_2: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.bfloat16)
        sum_2: "f32[1, 1, 16, 64]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 1], True, dtype = torch.float32);  convert_element_type_2 = None
        view_3: "f32[16, 64]" = torch.ops.aten.view.default(sum_2, _shape_param_3);  sum_2 = _shape_param_3 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        view_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute_4: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_4, [0, 1, 4, 2, 3]);  view_4 = None
        clone: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_5: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_5, 0);  view_5 = None
        return (view_1, view_3, squeeze_2)



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
