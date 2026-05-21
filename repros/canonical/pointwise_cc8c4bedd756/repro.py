"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: cc8c4bedd756
Shape hash: 94759a4e
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
    def forward(self, slice_1: "c64[32, 32]", view_as_complex_14: "c64[32, 32, 8, 32]", view_as_complex_15: "c64[32, 32, 8, 32]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "c64[1, 32, 1, 32]" = torch.ops.aten.reshape.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        mul_tensor: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_14, reshape_default);  view_as_complex_14 = None
        mul_tensor_1: "c64[32, 32, 8, 32]" = torch.ops.aten.mul.Tensor(view_as_complex_15, reshape_default);  view_as_complex_15 = reshape_default = None
        return (mul_tensor, mul_tensor_1)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
