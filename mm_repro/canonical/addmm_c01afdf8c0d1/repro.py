"""
Standalone repro captured via capture_hook.
Label: repros/models/torchbench/train/BERT_pytorch/full_graph_000.py
Pattern hash: c01afdf8c0d1
Shape hash: d9a512a1
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
    def forward(self, arg0_1: "f32[20005]", arg1_1: "f32[20005, 768]", arg2_1: "bf16[2048, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[20005]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[20005, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[768, 20005]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        constant_pad_nd: "bf16[768, 20008]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        full: "bf16[3]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        cat: "bf16[20008]" = torch.ops.aten.cat.default([convert_element_type, full]);  convert_element_type = full = None
        addmm: "bf16[2048, 20008]" = torch.ops.aten.addmm.default(cat, arg2_1, constant_pad_nd);  cat = arg2_1 = constant_pad_nd = None
        slice_1: "bf16[2048, 20005]" = torch.ops.aten.slice.Tensor(addmm, 1, 0, -3);  addmm = None
        view: "bf16[16, 128, 20005]" = torch.ops.aten.view.default(slice_1, _shape_param_2);  slice_1 = _shape_param_2 = None
        convert_element_type_2: "f32[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        return convert_element_type_2



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
