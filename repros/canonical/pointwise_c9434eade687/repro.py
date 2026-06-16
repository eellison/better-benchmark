"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_infer
Pattern hash: c9434eade687
Shape hash: 5545c8e0
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  arg1_1 = clone = None
        unsqueeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        unsqueeze_2: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_3: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        view_1: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_1);  unsqueeze_3 = _shape_param_1 = None
        squeeze_1: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        unsqueeze_4: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_5: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 4);  unsqueeze_4 = None
        view_2: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_5, _shape_param_2);  unsqueeze_5 = _shape_param_2 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_2, 0);  view_2 = None
        slice_1: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807)
        return (embedding, squeeze, squeeze_1, squeeze_2, slice_1)



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
