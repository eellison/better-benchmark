"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 8e602a7654f4
Shape hash: b3d88053
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
    def forward(self, arg0_1: "f32[30522, 128]", arg1_1: "i64[256, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        slice_1: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_1, [0, 0, 0, 1, 0, 0], 0.0);  slice_1 = None
        slice_2: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 1, 0, 0, 0], 0.0);  slice_2 = None
        cat: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        convert_element_type: "bf16[256, 128, 384]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        view: "bf16[32768, 384]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        return view



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
