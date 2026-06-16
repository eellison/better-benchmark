"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 3124e7b9efb0
Shape hash: c404fad8
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "bf16[32768, 512]", arg2_1: "f32[512, 512]", arg3_1: "f32[2, 512]", arg4_1: "f32[512]", arg5_1: "f32[512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 128);  arg0_1 = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        view: "bf16[256, 128, 512]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        embedding: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg2_1, slice_1);  arg2_1 = slice_1 = None
        embedding_1: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(arg3_1, full);  arg3_1 = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view, embedding);  view = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg4_1);  add_1 = arg4_1 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg5_1);  mul = arg5_1 = None
        convert_element_type: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_1: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        return (full, embedding, embedding_1, add_2, view_1)



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
