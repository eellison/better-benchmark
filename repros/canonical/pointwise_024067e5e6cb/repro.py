"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_000
Pattern hash: 024067e5e6cb
Shape hash: bb6b1620
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
_shapes_config = "(T([1, 512], i64, gen=Index(512)), T([32768, 512], f32), T([512, 512], f32), T([2, 512], f32), T([512], f32), T([512], f32), S([256, 128, 512]), S([32768, 512]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "i64[1, 512]", addmm: "f32[32768, 512]", arg5_1: "f32[512, 512]", arg6_1: "f32[2, 512]", arg7_1: "f32[512]", arg8_1: "f32[512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        full_default: "i64[256, 128]" = torch.ops.aten.full.default([256, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        embedding_default: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_tensor);  arg5_1 = slice_tensor = None
        embedding_default_1: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full_default);  arg6_1 = full_default = None
        add_tensor: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_default, embedding_default);  view_default = embedding_default = None
        add_tensor_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1);  add_tensor = embedding_default_1 = None
        mul_tensor: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_tensor_1, arg7_1);  add_tensor_1 = arg7_1 = None
        add_tensor_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_tensor, arg8_1);  mul_tensor = arg8_1 = None
        view_default_1: "f32[32768, 512]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        return view_default_1



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
