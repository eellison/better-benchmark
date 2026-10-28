"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/XLNetLMHeadModel/full_graph_000.py
Pattern hash: 2fce44d91a8a
Shape hash: a1c20a52
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
    def forward(self, arg0_1: "bf16[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        unsqueeze: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        permute: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_1: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_1: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute, permute_1);  permute = permute_1 = None
        sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_0);  unsqueeze_2 = _shape_param_0 = None
        clone: "f32[1024, 16, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
        convert_element_type_2: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone, torch.bfloat16);  clone = None
        unsqueeze_3: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        unsqueeze_4: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_1);  unsqueeze_4 = _shape_param_1 = None
        squeeze: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        unsqueeze_5: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 3);  arg0_1 = None
        unsqueeze_6: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view_1: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_2);  unsqueeze_6 = _shape_param_2 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        mm: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        unsqueeze_7: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_2: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_7, _shape_param_3);  unsqueeze_7 = _shape_param_3 = None
        permute_2: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_2, _shape_param_4);  permute_2 = _shape_param_4 = None
        unsqueeze_8: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_3, 4);  view_3 = None
        permute_3: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_8, [1, 2, 4, 0, 3]);  unsqueeze_8 = None
        permute_4: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_3, [0, 1, 4, 3, 2]);  permute_3 = None
        view_4: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_4, _shape_param_5);  permute_4 = _shape_param_5 = None
        return view_4



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
