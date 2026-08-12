"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train
Pattern hash: cf0dfeba0347
Shape hash: 2b53f84f
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
    def forward(self, arg0_1: "f32[50257, 2048]", arg1_1: "i64[32, 128]", arg2_1: "f32[2048, 2048]", arg3_1: "f32[2048]", arg4_1: "f32[2048]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding: "f32[32, 128, 2048]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        expand: "i64[32, 128]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_0);  _shape_param_0 = None
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        cat: "i64[32, 129]" = torch.ops.aten.cat.default([sub, expand], -1);  sub = expand = None
        slice_2: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 128)
        slice_3: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 129);  cat = None
        sub_1: "i64[32, 128]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None
        add_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_1, getitem_1)
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        convert_element_type: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        return (embedding, unsqueeze, ne, embedding_1, add_1, getitem_1, rsqrt, view)



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
