"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: 17476e33e2e0
Shape hash: ba0ea732
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
_shapes_config = "(T([1, 4096], i64), T([64, 1, 64], f16), T([1, 64, 192], f16), S([1, 64, 64, 64]), S([1, 64, 64, 192]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 4096]", arg1_1: "f16[64, 1, 64]", arg2_1: "f16[1, 64, 192]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:266 in forward, code: max_position_id = position_ids.max().item()
        max_default: "i64[]" = torch.ops.aten.max.default(arg0_1);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:227 in forward, code: weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        expand_default: "f16[1, 64, 64, 64]" = torch.ops.aten.expand.default(arg1_1, _shape_param_0);  arg1_1 = _shape_param_0 = None
        expand_default_1: "f16[1, 64, 64, 192]" = torch.ops.aten.expand.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        return (max_default, expand_default, expand_default_1)



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
