"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer
Pattern hash: 6ae114110db5
Shape hash: 3fee83c6
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
    def forward(self, arg0_1: "bf16[1, 3, 256, 256]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[1, 3, 256, 256]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        iota: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_1: "i64[262]" = torch.ops.aten.abs.default(iota);  iota = None
        sub: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_1);  abs_1 = None
        abs_2: "i64[262]" = torch.ops.aten.abs.default(sub);  sub = None
        sub_1: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_2);  abs_2 = None
        _unsafe_index: "f32[1, 3, 262, 256]" = torch.ops.aten._unsafe_index.Tensor(convert_element_type, [None, None, sub_1, None]);  convert_element_type = sub_1 = None
        iota_1: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_3: "i64[262]" = torch.ops.aten.abs.default(iota_1);  iota_1 = None
        sub_2: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_3);  abs_3 = None
        abs_4: "i64[262]" = torch.ops.aten.abs.default(sub_2);  sub_2 = None
        sub_3: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_4);  abs_4 = None
        _unsafe_index_1: "f32[1, 3, 262, 262]" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index, [None, None, None, sub_3]);  _unsafe_index = sub_3 = None
        convert_element_type_1: "bf16[1, 3, 262, 262]" = torch.ops.prims.convert_element_type.default(_unsafe_index_1, torch.bfloat16);  _unsafe_index_1 = None
        return convert_element_type_1



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
