"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/XLNetLMHeadModel/full_graph_001.py
Pattern hash: 499a003de4cb
Shape hash: 5246f0df
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 1023]", arg1_1: "bf16[256, 64, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        full: "bf16[16, 16, 1024, 512]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(full, view, 2, 1, 9223372036854775807);  full = view = None
        view_1: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(slice_scatter, _shape_param_2);  slice_scatter = _shape_param_2 = None
        view_2: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute, _shape_param_4);  permute = _shape_param_4 = None
        bmm: "bf16[256, 64, 1024]" = torch.ops.aten.bmm.default(arg1_1, view_3);  arg1_1 = view_3 = None
        view_4: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.view.default(bmm, _shape_param_5);  bmm = _shape_param_5 = None
        permute_1: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 1, 4, 3, 2]);  view_4 = None
        permute_2: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(permute_1, [3, 0, 1, 4, 2]);  permute_1 = None
        squeeze: "bf16[1024, 16, 16, 64]" = torch.ops.aten.squeeze.dim(permute_2, 4);  permute_2 = None
        view_5: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.view.default(squeeze, _shape_param_6);  squeeze = _shape_param_6 = None
        permute_3: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.permute.default(view_5, [0, 1, 4, 2, 3]);  view_5 = None
        clone: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_6: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(clone, _shape_param_7);  clone = _shape_param_7 = None
        squeeze_1: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_6, 0);  view_6 = None
        return squeeze_1



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
