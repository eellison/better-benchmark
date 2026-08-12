"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: f84d2f33f15b
Shape hash: d7517139
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
    def forward(self, _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        iota: "i64[6291456]" = torch.ops.prims.iota.default(6291456, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided: "i64[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(iota, _shape_param_0, _shape_param_1, 0);  iota = _shape_param_0 = _shape_param_1 = None
        clone: "i64[96, 3, 512, 64]" = torch.ops.aten.clone.default(as_strided, memory_format = torch.contiguous_format);  as_strided = None
        view: "i64[9437184]" = torch.ops.aten.view.default(clone, _shape_param_2);  clone = _shape_param_2 = None
        return view



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
