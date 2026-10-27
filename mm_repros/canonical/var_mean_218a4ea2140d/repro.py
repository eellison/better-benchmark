"""
Standalone repro captured via capture_hook.
Label: DistillGPT2
Pattern hash: 218a4ea2140d
Shape hash: f808d4da
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
    def forward(self, arg0_1: "bf16[50257, 768]", arg1_1: "i64[32, 512]", arg2_1: "bf16[1024, 768]", arg3_1: "bf16[768]", arg4_1: "bf16[768]", arg5_1: "bf16[2304]", arg6_1: "bf16[768, 2304]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
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
        addmm: "bf16[16384, 2304]" = torch.ops.aten.addmm.default(arg5_1, view, arg6_1);  arg5_1 = view = arg6_1 = None
        view_1: "bf16[32, 512, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        split = torch.ops.aten.split.Tensor(view_1, 768, 2);  view_1 = None
        getitem_2: "bf16[32, 512, 768]" = split[0]
        getitem_3: "bf16[32, 512, 768]" = split[1]
        getitem_4: "bf16[32, 512, 768]" = split[2];  split = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, _shape_param_2);  getitem_2 = _shape_param_2 = None
        permute: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_3, _shape_param_3);  getitem_3 = _shape_param_3 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(getitem_4, _shape_param_4);  getitem_4 = _shape_param_4 = None
        permute_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze, _shape_param_5);  unsqueeze = _shape_param_5 = None
        slice_1: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand, 1, 0, 1)
        sub_1: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        cat: "i64[32, 513]" = torch.ops.aten.cat.default([sub_1, expand], -1);  sub_1 = expand = None
        slice_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 513)
        slice_3: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 512);  cat = None
        sub_2: "i64[32, 512]" = torch.ops.aten.sub.Tensor(slice_2, slice_3);  slice_2 = slice_3 = None
        ne: "b8[32, 512]" = torch.ops.aten.ne.Scalar(sub_2, 1);  sub_2 = None
        return (add_1, permute, permute_1, permute_2, ne)



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
