"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_infer_000
Pattern hash: 06924cc70cb4
Shape hash: cc03aa16
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
_shapes_config = "(T([50257, 768], f32), T([32, 512], i64, gen=Index(50257)), T([1024, 768], f32), T([768], f32), T([768], f32), S([-1, 768]), S([32, -1]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50257, 768]", arg0_1: "i64[32, 512]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[512]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        embedding_default_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze_default);  arg2_1 = None
        add_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        add_tensor_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        add_tensor_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        view_default: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_0);  add_tensor_3 = _shape_param_0 = None
        expand_default: "i64[32, 512]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_1);  unsqueeze_default = _shape_param_1 = None
        slice_tensor: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor_1: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[32, 513]" = torch.ops.aten.cat.default([sub_tensor_1, expand_default], -1);  sub_tensor_1 = expand_default = None
        slice_tensor_1: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 513)
        slice_tensor_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 512);  cat_default = None
        sub_tensor_2: "i64[32, 512]" = torch.ops.aten.sub.Tensor(slice_tensor_1, slice_tensor_2);  slice_tensor_1 = slice_tensor_2 = None
        ne_scalar: "b8[32, 512]" = torch.ops.aten.ne.Scalar(sub_tensor_2, 1);  sub_tensor_2 = None
        return (view_default, ne_scalar)



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
