"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_infer
Pattern hash: fda6b7a40722
Shape hash: 7a2cdb11
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
    def forward(self, arg0_1: "bf16[50257, 768]", arg1_1: "i64[32, 512]", arg2_1: "bf16[1024, 768]", arg3_1: "bf16[768]", arg4_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[512]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        embedding_1: "bf16[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None
        add_1: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  convert_element_type_1 = _shape_param_0 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_1);  unsqueeze = _shape_param_1 = None
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub_1: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        cat: "i64[32, 513]" = torch.ops.aten.cat.default([sub_1, expand], -1);  sub_1 = expand = None
        slice_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_3: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
        sub_2: "i64[32, 512]" = torch.ops.aten.sub.Tensor(slice_2, slice_3);  slice_2 = slice_3 = None
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(sub_2, 1);  sub_2 = None
        return (add_1, view, ne)



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
