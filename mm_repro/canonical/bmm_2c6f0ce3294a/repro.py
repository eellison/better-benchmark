"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/DebertaV2ForMaskedLM/full_graph_001.py
Pattern hash: 2c6f0ce3294a
Shape hash: cc61df14
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
    def forward(self, arg0_1: "bf16[192, 512, 512]", arg1_1: "bf16[192, 512, 64]", arg2_1: "bf16[192, 512, 512]", arg3_1: "bf16[192, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        bmm: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        bmm_1: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        view: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm, _shape_param_0);  bmm = _shape_param_0 = None
        permute: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view, [0, 2, 1, 3]);  view = None
        clone: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        view_3: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_3);  bmm_1 = _shape_param_3 = None
        permute_2: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        clone_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view_4: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_1, _shape_param_4);  clone_1 = _shape_param_4 = None
        view_5: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[1536, 4096]" = torch.ops.aten.permute.default(view_5, [1, 0])
        return (view_2, permute_1, view_5, permute_3)



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
