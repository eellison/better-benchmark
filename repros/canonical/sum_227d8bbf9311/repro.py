"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: 227d8bbf9311
Shape hash: 3cf239db
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
    def forward(self, addmm: "f32[32, 128]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        pow_tensor_scalar: "f32[32, 128]" = torch.ops.aten.pow.Tensor_Scalar(addmm, 2.0)
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(pow_tensor_scalar, [1], True);  pow_tensor_scalar = None
        pow_tensor_scalar_1: "f32[32, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_dim_int_list, 0.5);  sum_dim_int_list = None
        clamp_min_default: "f32[32, 1]" = torch.ops.aten.clamp_min.default(pow_tensor_scalar_1, 1e-12);  pow_tensor_scalar_1 = None
        expand_default: "f32[32, 128]" = torch.ops.aten.expand.default(clamp_min_default, _shape_param_0);  clamp_min_default = _shape_param_0 = None
        div_tensor: "f32[32, 128]" = torch.ops.aten.div.Tensor(addmm, expand_default);  addmm = expand_default = None
        return div_tensor


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
