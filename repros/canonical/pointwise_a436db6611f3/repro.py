"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer
Pattern hash: a436db6611f3
Shape hash: ffd9ed70
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
    def forward(self, arg269_1: "f32[32, 128]", arg268_1: "i64[32]", wait_tensor: "f32[32, 128]", arg270_1: "f32[128, 32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 1, 128]" = torch.ops.aten.reshape.default(arg269_1, _shape_param_0);  _shape_param_0 = None
        reshape_default_1: "i64[1, 32]" = torch.ops.aten.reshape.default(arg268_1, [1, -1]);  arg268_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(reshape_default_1, 0, 0);  reshape_default_1 = None
        index_tensor: "f32[32, 128]" = torch.ops.aten.index.Tensor(wait_tensor, [select_int]);  wait_tensor = select_int = None
        reshape_default_2: "f32[32, 128, 1]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None
        unsqueeze_default: "f32[32, 128, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, 2);  arg269_1 = None
        reshape_default_3: "f32[1, 32, 128]" = torch.ops.aten.reshape.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None
        squeeze_dim: "f32[32, 128]" = torch.ops.aten.squeeze.dim(reshape_default_3, 0);  reshape_default_3 = None
        unsqueeze_default_1: "f32[128, 32000, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, 2);  arg270_1 = None
        reshape_default_4: "f32[1, 128, 32000]" = torch.ops.aten.reshape.default(unsqueeze_default_1, _shape_param_3);  unsqueeze_default_1 = _shape_param_3 = None
        squeeze_dim_1: "f32[128, 32000]" = torch.ops.aten.squeeze.dim(reshape_default_4, 0);  reshape_default_4 = None
        return (reshape_default, reshape_default_2, squeeze_dim, squeeze_dim_1)


def _default_make_inputs():
    return []


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
