"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/DebertaV2ForMaskedLM/full_graph_001.py
Pattern hash: 715e9839297e
Shape hash: 9e06e110
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
    def forward(self, arg0_1: "bf16[192, 64, 512]", arg1_1: "bf16[192, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 64, 512]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm, full);  bmm = full = None
        permute: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        view: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        clone: "bf16[8, 512, 1536]" = torch.ops.aten.clone.default(view_1, memory_format = torch.contiguous_format);  view_1 = None
        view_2: "bf16[4096, 1536]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_2: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_2, permute_2)



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
