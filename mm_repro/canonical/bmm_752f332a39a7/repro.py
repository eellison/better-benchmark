"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/XLNetLMHeadModel/full_graph_001.py
Pattern hash: 752f332a39a7
Shape hash: f56989bb
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 1023]", arg1_1: "bf16[256, 1024, 64]", arg2_1: "bf16[256, 512, 512]", arg3_1: "bf16[256, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        full: "bf16[16, 16, 1024, 512]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full, view, 2, 1, 9223372036854775807);  full = view = None
        view_1: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(slice_scatter, _shape_param_2);  slice_scatter = _shape_param_2 = None
        view_2: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute, _shape_param_4);  permute = _shape_param_4 = None
        bmm: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_3, arg1_1);  view_3 = arg1_1 = None
        view_4: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm, _shape_param_5);  bmm = _shape_param_5 = None
        permute_1: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_4, [0, 1, 2, 4, 3]);  view_4 = None
        permute_2: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1, [2, 0, 1, 4, 3]);  permute_1 = None
        squeeze: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2, 4);  permute_2 = None
        convert_element_type: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze, torch.float32);  squeeze = None
        convert_element_type_1: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.bfloat16)
        bmm_1: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view_5: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.view.default(bmm_1, _shape_param_6);  bmm_1 = _shape_param_6 = None
        permute_3: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(view_5, [0, 1, 2, 4, 3]);  view_5 = None
        permute_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_3, [2, 0, 1, 4, 3]);  permute_3 = None
        squeeze_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_4, 4);  permute_4 = None
        convert_element_type_2: "f32[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        convert_element_type_3: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.bfloat16);  convert_element_type_2 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(convert_element_type_1, convert_element_type_3);  convert_element_type_1 = convert_element_type_3 = None
        view_6: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.view.default(add, _shape_param_7);  add = _shape_param_7 = None
        permute_5: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_6, [0, 1, 4, 2, 3]);  view_6 = None
        clone: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_7: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone, _shape_param_8);  clone = _shape_param_8 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_7, 0);  view_7 = None
        return (convert_element_type, squeeze_2)



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
