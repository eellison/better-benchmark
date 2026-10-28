"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/XLNetLMHeadModel/full_graph_000.py
Pattern hash: d9c22485d165
Shape hash: 56d7afc0
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", arg2_1: "bf16[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        unsqueeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3);  embedding = None
        unsqueeze_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        unsqueeze_2: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_3: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        view_1: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_1);  unsqueeze_3 = _shape_param_1 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        unsqueeze_4: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_2);  unsqueeze_4 = _shape_param_2 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        unsqueeze_5: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_3, 4);  view_3 = None
        permute_2: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_5, [4, 1, 2, 3, 0]);  unsqueeze_5 = None
        permute_3: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2, [1, 2, 4, 3, 0]);  permute_2 = None
        view_4: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        return view_4



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
