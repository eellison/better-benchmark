"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_infer_000
Pattern hash: 51020b36ae3c
Shape hash: 9565ef11
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
_shapes_config = "(T([30522, 128], f32), T([256, 128], i64, gen=Index(30522)), S([32768, 384]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[30522, 128]", arg0_1: "i64[256, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding_default: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None
        slice_tensor: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807)
        constant_pad_nd_default: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor = None
        slice_tensor_1: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1)
        constant_pad_nd_default_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_1 = None
        cat_default: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None
        view_default: "f32[32768, 384]" = torch.ops.aten.view.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
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
