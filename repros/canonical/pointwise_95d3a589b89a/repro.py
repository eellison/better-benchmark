"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: 95d3a589b89a
Shape hash: 68f9fa5a
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
_shapes_config = "(T([1024, 1536], f32), T([1024, 1536], f32), S([32, 32, 1536]), S([32, 32, 1536]), S([1024, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, mm_53: "f32[1024, 1536]", mm_54: "f32[1024, 1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_53, _shape_param_0);  mm_53 = _shape_param_0 = None
        neg_default: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        reshape_default_1: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_54, _shape_param_1);  mm_54 = _shape_param_1 = None
        mul_tensor: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_tensor, reshape_default_1);  div_tensor = reshape_default_1 = None
        reshape_default_2: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        return reshape_default_2



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
