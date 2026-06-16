"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_train
Pattern hash: 645fb123c324
Shape hash: 59aa3e3a
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
    def forward(self, arg0_1: "f32[514, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        iota: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        expand: "i64[64, 256]" = torch.ops.aten.expand.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        add: "i64[64, 256]" = torch.ops.aten.add.Tensor(expand, 2);  expand = None
        embedding: "f32[64, 256, 1024]" = torch.ops.aten.embedding.default(arg0_1, add);  arg0_1 = None
        return (add, embedding)



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
