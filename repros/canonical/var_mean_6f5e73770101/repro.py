"""
Standalone repro captured via capture_hook.
Label: hf_DistilBertForMaskedLM_train_000
Pattern hash: 6f5e73770101
Shape hash: 1106a851
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
_shapes_config = "(T([30522, 768], f32), T([256, 128], i64, gen=Index(30522)), T([1, 512], i64, gen=Index(512)), T([512, 768], f32), T([768], f32), T([768], f32), S([32768, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[30522, 768]", arg0_1: "i64[256, 128]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        embedding_default: "f32[256, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        embedding_default_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_tensor);  arg3_1 = slice_tensor = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1);  embedding_default = embedding_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[256, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        inductor_seeds_default: "i64[13]" = torch.ops.prims.inductor_seeds.default(13, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 128, 768]" = torch.ops.prims.inductor_random.default([256, 128, 768], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[256, 128, 768]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_2);  gt_scalar = add_tensor_2 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        view_default: "f32[32768, 768]" = torch.ops.aten.view.default(mul_tensor_3, _shape_param_0);  mul_tensor_3 = _shape_param_0 = None
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
