"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GPTJForQuestionAnswering/full_graph_000.py
Pattern hash: 585257e60d0b
Shape hash: 10b53b64
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
    def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2, 4096]", arg2_1: "bf16[128, 4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[2, 4096]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[4096, 2]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        constant_pad_nd: "bf16[4096, 8]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        full: "bf16[6]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        cat: "bf16[8]" = torch.ops.aten.cat.default([convert_element_type, full]);  convert_element_type = full = None
        addmm: "bf16[128, 8]" = torch.ops.aten.addmm.default(cat, arg2_1, constant_pad_nd);  cat = arg2_1 = constant_pad_nd = None
        slice_1: "bf16[128, 2]" = torch.ops.aten.slice.Tensor(addmm, 1, 0, -6);  addmm = None
        view: "bf16[1, 128, 2]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        split = torch.ops.aten.split.Tensor(view, 1, -1);  view = None
        getitem: "bf16[1, 128, 1]" = split[0]
        getitem_1: "bf16[1, 128, 1]" = split[1];  split = None
        squeeze: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem, -1);  getitem = None
        clone: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        squeeze_1: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem_1, -1);  getitem_1 = None
        clone_1: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        convert_element_type_2: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone, torch.float32)
        convert_element_type_3: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone_1, torch.float32)
        return (clone, clone_1, convert_element_type_2, convert_element_type_3)



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
