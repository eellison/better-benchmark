"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: aa2f84e0b60d
Shape hash: f164ab61
"""
import sys
from pathlib import Path

import sys
from pathlib import Path
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_53: "f32[1024, 1536]", mm_54: "f32[1024, 1536]", arg90_1: "f32[512, 1536]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_53, _shape_param_0);  mm_53 = _shape_param_0 = None
        neg_default: "f32[32, 32, 1536]" = torch.ops.aten.neg.default(reshape_default)
        exp_default: "f32[32, 32, 1536]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 32, 1536]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 32, 1536]" = torch.ops.aten.div.Tensor(reshape_default, add_tensor);  reshape_default = add_tensor = None
        reshape_default_1: "f32[32, 32, 1536]" = torch.ops.aten.reshape.default(mm_54, _shape_param_1);  mm_54 = _shape_param_1 = None
        mul_tensor: "f32[32, 32, 1536]" = torch.ops.aten.mul.Tensor(div_tensor, reshape_default_1);  div_tensor = reshape_default_1 = None
        reshape_default_2: "f32[1024, 1536]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        permute_default: "f32[1536, 512]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        return (reshape_default_2, permute_default)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
