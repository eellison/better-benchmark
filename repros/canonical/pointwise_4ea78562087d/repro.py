"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer
Pattern hash: 4ea78562087d
Shape hash: 25f363be
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
_shapes_config = "(T([1, 3, 256, 256], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 3, 256, 256]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        iota_default: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default: "i64[262]" = torch.ops.aten.abs.default(iota_default);  iota_default = None
        sub_tensor: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default);  abs_default = None
        abs_default_1: "i64[262]" = torch.ops.aten.abs.default(sub_tensor);  sub_tensor = None
        sub_tensor_1: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_1);  abs_default_1 = None
        _unsafe_index_tensor: "f32[1, 3, 262, 256]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, sub_tensor_1, None]);  arg0_1 = sub_tensor_1 = None
        iota_default_1: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default_2: "i64[262]" = torch.ops.aten.abs.default(iota_default_1);  iota_default_1 = None
        sub_tensor_2: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_2);  abs_default_2 = None
        abs_default_3: "i64[262]" = torch.ops.aten.abs.default(sub_tensor_2);  sub_tensor_2 = None
        sub_tensor_3: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_3);  abs_default_3 = None
        _unsafe_index_tensor_1: "f32[1, 3, 262, 262]" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_tensor, [None, None, None, sub_tensor_3]);  _unsafe_index_tensor = sub_tensor_3 = None
        return _unsafe_index_tensor_1



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
