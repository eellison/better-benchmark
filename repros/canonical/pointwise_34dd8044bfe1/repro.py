"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_005
Pattern hash: 34dd8044bfe1
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

_repro_version = 2
_shapes_config = "(S([12, 64, 64]))"

class Repro(torch.nn.Module):
    def forward(self, _shape_param_0):
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[12, 64, 1, 64]" = torch.ops.prims.inductor_random.default([12, 64, 1, 64], inductor_lookup_seed_default, 'randn');  inductor_lookup_seed_default = None
        unsqueeze_default: "f32[12, 64, 1, 64, 1]" = torch.ops.aten.unsqueeze.default(inductor_random_default, 4);  inductor_random_default = None
        unsqueeze_default_1: "f32[12, 64, 1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 5);  unsqueeze_default = None
        view_default: "f32[12, 64, 64]" = torch.ops.aten.view.default(unsqueeze_default_1, _shape_param_0);  unsqueeze_default_1 = _shape_param_0 = None
        return view_default



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
