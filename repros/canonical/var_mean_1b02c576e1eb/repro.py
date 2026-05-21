"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_000
Pattern hash: 1b02c576e1eb
Shape hash: d6f69916
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
_shapes_config = "(T([50257, 2048], f32), T([32, 128], i64, gen=Index(50257)), T([2048, 2048], f32), T([2048], f32), T([2048], f32), S([32, -1]), S([4096, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[50257, 2048]", arg0_1: "i64[32, 128]", arg2_1: "f32[2048, 2048]", arg3_1: "f32[2048]", arg4_1: "f32[2048]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        embedding_default: "f32[32, 128, 2048]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor: "i64[128]" = torch.ops.aten.add.Tensor(iota_default, 0);  iota_default = None
        unsqueeze_default: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        expand_default: "i64[32, 128]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  _shape_param_0 = None
        slice_tensor: "i64[32, 1]" = torch.ops.aten.slice.Tensor(expand_default, 1, 0, 1)
        sub_tensor: "i64[32, 1]" = torch.ops.aten.sub.Tensor(slice_tensor, 1);  slice_tensor = None
        cat_default: "i64[32, 129]" = torch.ops.aten.cat.default([sub_tensor, expand_default], -1);  sub_tensor = expand_default = None
        slice_tensor_1: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 0, 128)
        slice_tensor_2: "i64[32, 128]" = torch.ops.aten.slice.Tensor(cat_default, -1, 1, 129);  cat_default = None
        sub_tensor_1: "i64[32, 128]" = torch.ops.aten.sub.Tensor(slice_tensor_2, slice_tensor_1);  slice_tensor_2 = slice_tensor_1 = None
        ne_scalar: "b8[32, 128]" = torch.ops.aten.ne.Scalar(sub_tensor_1, 1);  sub_tensor_1 = None
        embedding_default_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze_default);  arg2_1 = unsqueeze_default = None
        add_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1);  add_tensor_1 = getitem_1 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        add_tensor_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        view_default: "f32[4096, 2048]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_1);  add_tensor_3 = _shape_param_1 = None
        return (ne_scalar, view_default)



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
