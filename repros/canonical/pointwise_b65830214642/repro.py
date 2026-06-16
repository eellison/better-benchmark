"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_infer
Pattern hash: b65830214642
Shape hash: 981155f5
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
    def forward(self, arg0_1: "bf16[4096, 1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        clone: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_2: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        permute_1: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_1, full);  permute_1 = full = None
        return div



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
