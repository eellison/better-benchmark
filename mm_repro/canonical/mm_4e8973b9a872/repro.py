"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/GPT2ForSequenceClassification/full_graph_000.py
Pattern hash: 4e8973b9a872
Shape hash: 217b5f62
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
    def forward(self, arg0_1: "bf16[2, 768]", arg1_1: "bf16[8192, 768]", arg2_1: "i64[8]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "bf16[768, 2]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[8192, 2]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[8, 1024, 2]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[8, 2]" = torch.ops.aten.index.Tensor(view, [iota, arg2_1]);  view = iota = arg2_1 = None
        convert_element_type: "f32[8, 2]" = torch.ops.prims.convert_element_type.default(index, torch.float32)
        return (index, convert_element_type)



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
